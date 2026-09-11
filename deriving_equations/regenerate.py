"""Regenerate the machine-written blocks of the .pyx files

The .pyx sources are mostly expressions printed by the scripts under
``donnell/`` and ``sanders/``. Keeping that link manual is how
``update_KC0`` of the Sanders element ended up missing a term that its own
derivation script had been printing for a while. This script closes the
loop: it runs each derivation, picks the printed blocks apart, and rewrites
the matching region of the .pyx in place.

Usage
-----
    python deriving_equations/regenerate.py --check
    python deriving_equations/regenerate.py --write

``--check`` reports which blocks are out of date without touching anything,
``--write`` rewrites them. Run ``--check`` again afterwards; it must come
back clean. The .pyx still has to be cythonised and compiled after a
``--write``.

The blocks are found by their shape rather than by markers, because that is
what the derivation scripts already emit:

``assign``   lines such as ``Bm1_01 = ...``, one per nonzero entry of a
             strain-displacement operator, indented inside the integration
             loop
``values``   the ``KC0v[k] += ...`` / ``k += 1`` pairs, the body of the
             element matrix
``rowcol``   the ``KC0r[k] = ...`` / ``KC0c[k] = ...`` pairs that place the
             sparse entries
``stress``   the ``Nxx = ...`` .. ``Mxy = ...`` lines of update_fint
``fint``     the ``fint[i + c1] += ...`` lines
"""
import argparse
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# kinematics -> (derivation sub-directory, .pyx file)
TARGETS = {
    'donnell': ('donnell', os.path.join(ROOT, 'bfsccylinder', 'bfsccylinder.pyx')),
    'sanders': ('sanders', os.path.join(ROOT, 'bfsccylinder', 'bfsccylinder_sanders.pyx')),
}

# .pyx function -> derivation script that writes its blocks
SOURCES = [
    ('update_KC0', 'derive_bfsccylinder_KC0_sparse.py', 'KC0'),
    ('update_KCNL', 'derive_bfsccylinder_KC0L_KCL0_KCLL_sparse.py', 'KCNL'),
    ('update_KG', 'derive_bfsccylinder_KG_sparse.py', 'KG'),
    ('update_M', 'derive_bfsccylinder_M_sparse.py', 'M'),
    ('update_fint', 'derive_bfsccylinder_fint.py', None),
]

OPERATORS = ('Bm', 'BmL', 'Bb', 'G', 'Nu', 'Nv', 'Nw', 'Su', 'Sv', 'Sw')
_assign = re.compile(r'^(%s)(\d)_(\d\d) = ' % '|'.join(OPERATORS))
_stress = re.compile(r'^(Nxx|Nyy|Nxy|Mxx|Myy|Mxy) = ')
_size = re.compile(r'^(\w+)_SPARSE_SIZE (\d+)$')
_size_decl = re.compile(r'^(\w+)_SPARSE_SIZE = (\d+)$')


class Drift(Exception):
    """The .pyx and the derivations no longer line up structurally.

    Raised instead of skipping quietly, so that --check can never pass by
    simply failing to look at something.
    """


def run_derivation(path):
    """Run a derivation script and return its stdout as a list of lines."""
    out = subprocess.run([sys.executable, path], capture_output=True,
                         text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError('%s failed:\n%s' % (path, out.stderr[-4000:]))
    return out.stdout.replace('\r\n', '\n').split('\n')


def function_span(lines, name):
    """Line range [start, end) of a cpdef function body in the .pyx."""
    start = None
    for i, line in enumerate(lines):
        if line.startswith('cpdef ') and (' %s(' % name) in line:
            start = i
        elif start is not None and line.startswith('cpdef '):
            return start, i
    if start is None:
        raise KeyError(name)
    return start, len(lines)


def assign_blocks(lines, lo, hi):
    """Group consecutive ``Xn_nn = ...`` lines by operator name."""
    blocks = {}
    i = lo
    while i < hi:
        m = _assign.match(lines[i].strip())
        if m is None:
            i += 1
            continue
        op = m.group(1)
        j = i
        while j < hi:
            m2 = _assign.match(lines[j].strip())
            if m2 is None or m2.group(1) != op:
                break
            j += 1
        blocks.setdefault(op, []).append((i, j))
        i = j
    return blocks


def value_block(lines, lo, hi, tag):
    """Span of the ``<tag>v[k] += ...`` body, including its ``k += 1`` lines."""
    pat = re.compile(r'^%sv\[k\] \+=' % tag)
    idx = [i for i in range(lo, hi) if pat.match(lines[i].strip())]
    if not idx:
        return None
    return idx[0], idx[-1] + 1


def rowcol_block(lines, lo, hi, tag):
    pat = re.compile(r'^%s[rc]\[k\] =' % tag)
    idx = [i for i in range(lo, hi) if pat.match(lines[i].strip())]
    if not idx:
        return None
    return idx[0], idx[-1] + 1


def stress_block(lines, lo, hi):
    idx = [i for i in range(lo, hi) if _stress.match(lines[i].strip())]
    if not idx:
        return None
    return idx[0], idx[-1] + 1


def fint_block(lines, lo, hi):
    idx = [i for i in range(lo, hi) if lines[i].strip().startswith('fint[')]
    if not idx:
        return None
    return idx[0], idx[-1] + 1


def gen_assigns(gen, op):
    """Generated ``op`` assignment lines, in order, de-duplicated."""
    seen = {}
    order = []
    for line in gen:
        m = _assign.match(line.strip())
        if m is None or m.group(1) != op:
            continue
        key = '%s%s_%s' % (m.group(1), m.group(2), m.group(3))
        if key not in seen:
            order.append(key)
        seen[key] = line.strip()
    return [seen[k] for k in order]


def gen_values(gen, tag):
    pat = re.compile(r'^%sv\[k\] \+=' % tag)
    return [ln.strip() for ln in gen if pat.match(ln.strip())]


def gen_rowcols(gen, tag):
    pat = re.compile(r'^%s[rc]\[k\] =' % tag)
    return [ln.strip() for ln in gen if pat.match(ln.strip())]


def gen_stress(gen):
    out = {}
    for line in gen:
        m = _stress.match(line.strip())
        if m is not None:
            out[m.group(1)] = line.strip()
    return [out[k] for k in ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy')
            if k in out]


def gen_fint(gen):
    return [ln.strip() for ln in gen if ln.strip().startswith('fint[')]


def gen_sparse_size(gen, tag):
    """The entry count a derivation reports, e.g. "KCNL_SPARSE_SIZE 1024"."""
    for line in gen:
        m = _size.match(line.strip())
        if m is not None and m.group(1) == tag:
            return int(m.group(2))
    return None


def sparse_size_line(lines, tag):
    """Index of the module-level "<TAG>_SPARSE_SIZE = n" declaration."""
    idx = [i for i, ln in enumerate(lines)
           if _size_decl.match(ln.strip())
           and _size_decl.match(ln.strip()).group(1) == tag]
    if len(idx) != 1:
        return None
    return idx[0]


def declared_names(lines, lo, hi):
    """Every name declared by a cdef inside one function body."""
    names = set()
    for i in range(lo, hi):
        s = lines[i].strip()
        if not s.startswith('cdef '):
            continue
        parts = s.split(None, 2)
        if len(parts) < 3:
            continue
        for nm in parts[2].split(','):
            names.add(nm.strip().lstrip('*').split('[')[0].strip())
    return names


def indent_of(line):
    return line[:len(line) - len(line.lstrip())]


def interleave(body, pad, sep='k += 1'):
    """Emit ``expr`` / ``k += 1`` / ``expr`` ... the way the .pyx has it."""
    out = []
    for n, expr in enumerate(body):
        if n:
            out.append(pad + sep)
        out.append(pad + expr)
    return out


def interleave_pairs(pairs, pad, sep='k += 1'):
    """Emit r/c pairs separated by ``k += 1``."""
    out = []
    for n in range(0, len(pairs), 2):
        if n:
            out.append(pad + sep)
        out.append(pad + pairs[n])
        out.append(pad + pairs[n + 1])
    return out


class Edit(object):
    def __init__(self, label, lo, hi, new):
        self.label, self.lo, self.hi, self.new = label, lo, hi, new

    @property
    def changed(self):
        return self.old != self.new

    def describe(self):
        return '%s: lines %d-%d, %d -> %d lines' % (
            self.label, self.lo + 1, self.hi, len(self.old), len(self.new))


def plan(kin, verbose=False):
    sub, pyx_path = TARGETS[kin]
    text = io.open(pyx_path, encoding='utf8').read()
    newline = '\r\n' if '\r\n' in text else '\n'
    lines = text.replace('\r\n', '\n').split('\n')
    edits = []

    def require(want, have, what, fname):
        """A block the derivation writes must exist in the .pyx.

        Skipping it instead would let --check pass over a function that has
        drifted or was never complete, which is the one thing this tool is
        supposed to make impossible.
        """
        if want and have is None:
            raise Drift('%s: %s has no %s block, but %s writes %d lines of '
                        'one. Restore the block by hand, or remove the entry '
                        'from SOURCES.'
                        % (kin, fname, what, script, len(want)))
        return want and have

    for fname, script, tag in SOURCES:
        path = os.path.join(HERE, sub, script)
        if not os.path.exists(path):
            raise Drift('%s: derivation script is missing: %s' % (kin, path))
        try:
            lo, hi = function_span(lines, fname)
        except KeyError:
            raise Drift('%s: %s is not defined in %s, but %s generates its '
                        'code' % (kin, fname, pyx_path, script))
        gen = run_derivation(path)

        # strain-displacement operator assignments. The last group of a given
        # operator is the one inside the integration loop; anything above it
        # belongs to the hand-written class methods
        blocks = assign_blocks(lines, lo, hi)
        loop = dict((op, spans[-1]) for op, spans in blocks.items())
        needed = []
        for op in OPERATORS:
            want = gen_assigns(gen, op)
            if not want:
                continue
            needed.extend(w.split(' = ')[0] for w in want)
            if op in loop:
                a, b = loop[op]
                pad = indent_of(lines[a])
                edits.append(Edit('%s/%s %s' % (kin, fname, op), a, b,
                                  [pad + w for w in want]))
            elif loop:
                # the derivation started emitting an operator this function
                # did not carry before: put it after the last block
                at = max(b for _, b in loop.values())
                pad = indent_of(lines[min(a for a, _ in loop.values())])
                edits.append(Edit('%s/%s %s (new block)' % (kin, fname, op),
                                  at, at, [''] + [pad + w for w in want]))

        # every generated symbol must be declared, whether it came from a new
        # operator or was added to one that already had a block. An
        # undeclared local is assigned inside "with nogil", so Cython refuses
        # to compile it and regeneration leaves an unbuildable .pyx
        missing = [n for n in needed if n not in declared_names(lines, lo, hi)]
        if missing:
            decl = [i for i in range(lo, hi)
                    if re.match(r'^cdef double (%s)\d_\d\d'
                                % '|'.join(OPERATORS), lines[i].strip())]
            at = (decl[-1] + 1) if decl else (lo + 1)
            dpad = indent_of(lines[decl[-1]] if decl else lines[lo + 1])
            edits.append(Edit('%s/%s cdef' % (kin, fname), at, at,
                              [dpad + 'cdef double ' + ', '.join(missing)]))

        if tag is not None:
            want = gen_values(gen, tag)
            have = value_block(lines, lo, hi, tag)
            if require(want, have, 'values', fname):
                a, b = have
                pad = indent_of(lines[a])
                edits.append(Edit('%s/%s values' % (kin, fname), a, b,
                                  interleave(want, pad)))
            want = gen_rowcols(gen, tag)
            have = rowcol_block(lines, lo, hi, tag)
            if require(want, have, 'row/column', fname):
                a, b = have
                pad = indent_of(lines[a])
                edits.append(Edit('%s/%s rowcol' % (kin, fname), a, b,
                                  interleave_pairs(want, pad)))

            # the number of sparse entries the loop above writes. Callers size
            # their arrays as <TAG>_SPARSE_SIZE*num_elements, and the .pyx is
            # compiled with boundscheck=False, so a size left behind by a
            # derivation that gained or lost a nonzero entry is a silent
            # out-of-bounds write rather than an IndexError
            size = gen_sparse_size(gen, tag)
            if size is not None:
                at = sparse_size_line(lines, tag)
                if at is None:
                    raise Drift('%s: %s emits %s_SPARSE_SIZE but %s declares '
                                'no such constant' % (kin, script, tag, pyx_path))
                edits.append(Edit('%s/%s_SPARSE_SIZE' % (kin, tag), at, at + 1,
                                  ['%s_SPARSE_SIZE = %d' % (tag, size)]))
        else:
            want = gen_stress(gen)
            have = stress_block(lines, lo, hi)
            if require(want, have, 'stress resultant', fname):
                a, b = have
                pad = indent_of(lines[a])
                edits.append(Edit('%s/%s stress' % (kin, fname), a, b,
                                  [pad + w for w in want]))
            want = gen_fint(gen)
            have = fint_block(lines, lo, hi)
            if require(want, have, 'fint', fname):
                a, b = have
                pad = indent_of(lines[a])
                edits.append(Edit('%s/%s fint' % (kin, fname), a, b,
                                  [pad + w for w in want]))

    for e in edits:
        e.old = lines[e.lo:e.hi]
    return pyx_path, lines, newline, edits


def apply_edits(lines, edits):
    out = list(lines)
    for e in sorted(edits, key=lambda e: e.lo, reverse=True):
        out[e.lo:e.hi] = e.new
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true',
                    help='rewrite the .pyx files (default is a dry run)')
    ap.add_argument('--check', action='store_true',
                    help='exit non-zero if any block is out of date')
    ap.add_argument('--kinematics', choices=sorted(TARGETS), action='append',
                    help='restrict to one kinematics (default: both)')
    args = ap.parse_args()

    try:
        return run(args)
    except Drift as exc:
        print('DRIFT: %s' % exc)
        return 2


def run(args):
    stale = 0
    for kin in (args.kinematics or sorted(TARGETS)):
        pyx_path, lines, newline, edits = plan(kin)
        changed = [e for e in edits if e.changed]
        print('%s: %d generated blocks, %d out of date'
              % (kin, len(edits), len(changed)))
        for e in changed:
            print('    %s' % e.describe())
        stale += len(changed)
        if args.write and changed:
            new = apply_edits(lines, edits)
            io.open(pyx_path, 'w', encoding='utf8', newline='').write(
                newline.join(new))
            print('    wrote %s' % pyx_path)

    if args.check and stale:
        print('\n%d block(s) out of date; run with --write' % stale)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
