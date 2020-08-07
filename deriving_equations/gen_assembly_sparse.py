DOF = 10

for ci in ['c1', 'c2', 'c3', 'c4']:
    for i in range(DOF):
        for cj in ['c1', 'c2', 'c3', 'c4']:
            for j in range(DOF):
                print('k += 1')
                print('Ar[k] = %d + %s' % (i, ci))
                print('Ac[k] = %d + %s' % (j, cj))
                print('Av[k] = Ae[%d*DOF + %d, %d*DOF + %d]' % (
                    int(ci[-1])-1, i, int(cj[-1])-1, j))

