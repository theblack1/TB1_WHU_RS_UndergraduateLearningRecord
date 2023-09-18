def Mul(A, B):
    if len(A[0]) == len(B):
        res = [[0] * len(B[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    res[i][j] += A[i][k] * B[k][j]
        return res
    return ('输入矩阵有误！')

def Convar(ob1, ob2):
    n = len(ob1)
    if (n != len(ob2)):
        print('error! not the same number!')
        exit()

    mean1 = sum(ob1) / n
    mean2 = sum(ob2) / n

    convar = 0
    for i in range(n):
        convar += (ob1[i] - mean1) * (ob2[i] - mean2)

    return convar / n


if __name__ == '__main__':
    Ky = [[3, -2, 1]]
    Kz = [[-1], [-1], [4]]


    # 输入
    ob = str.split(input())
    ob = [float(i) for i in ob]

    # 截取
    x1 = ob[0:3]
    x2 = ob[3:6]
    x3 = ob[6:]

    # Dxx
    Dxx = [[Convar(x1, x1), Convar(x1, x2), Convar(x1, x3)],
           [Convar(x2, x1), Convar(x2, x2), Convar(x2, x3)],
           [Convar(x3, x1), Convar(x3, x2), Convar(x3, x3)]]

    # FDK
    FD = Mul(Ky, Dxx)
    # Fd = FD.copy()
    FDK = Mul(FD, Kz)

    # 输出
    print('{:.6f}'.format(FDK[0][0]))
    if Convar(x1, x2) == 0 and Convar(x3, x2) == 0 and Convar(x1, x3) == 0:
        print('YES')
    else:
        print('NO')

