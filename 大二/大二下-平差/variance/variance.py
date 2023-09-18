n = float(input())
ob0 =str.split(input())
ob = [float(i) for i in ob0]
mean = sum(ob)/n
var = 0
for i in ob:
    var += (i-mean)*(i-mean)

var = var/n

print('{:.6f}'.format(var))

