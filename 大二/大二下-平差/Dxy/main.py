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
    ob1 = str.split(input())
    ob1 = [float(i) for i in ob1]

    ob2 = str.split(input())
    ob2 = [float(i) for i in ob2]



    var1 = Convar(ob1, ob1)
    var2 = Convar(ob2, ob2)
    convar = Convar(ob2, ob1)

    print('{:.6f} {:.6f}\n{:.6f} {:.6f}'.format(var1, convar, convar, var2), end="")
