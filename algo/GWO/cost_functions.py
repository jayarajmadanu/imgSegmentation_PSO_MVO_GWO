import math
import random
import numpy as np


def getFunction(F):
    list = []
    if F == 'F0':
        list.append(F0)
        list.append(0)
        list.append(255)
        list.append(1)
    elif F == 'F1':
        list.append(F1)
        list.append(-100)
        list.append(100)
        list.append(10)
    elif F == 'F2':
        list.append(F2)
        list.append(-10)
        list.append(10)
        list.append(10)
    elif F == 'F3':
        list.append(F3)
        list.append(-100)
        list.append(100)
        list.append(10)
    elif F == 'F4':
        list.append(F4)
        list.append(-100)
        list.append(100)
        list.append(10)
    elif F == 'F5':
        list.append(F5)
        list.append(-30)
        list.append(30)
        list.append(10)
    elif F == 'F6':
        list.append(F6)
        list.append(-100)
        list.append(100)
        list.append(10)
    elif F == 'F7':
        list.append(F7)
        list.append(-1.28)
        list.append(1.28)
        list.append(10)
    elif F == 'F8':
        list.append(F8)
        list.append(-500)
        list.append(500)
        list.append(10)
    elif F == 'F9':
        list.append(F9)
        list.append(-5.12)
        list.append(5.12)
        list.append(10)
    elif F == 'F10':
        list.append(F10)
        list.append(-32)
        list.append(32)
        list.append(10)
    elif F == 'F11':
        list.append(F11)
        list.append(-600)
        list.append(600)
        list.append(10)
    elif F == 'F12':
        list.append(F12)
        list.append(-50)
        list.append(50)
        list.append(10)
    elif F == 'F13':
        list.append(F13)
        list.append(-50)
        list.append(50)
        list.append(10)
    return list


def F0(x, H):
    l = 0
    x = x.tolist()
    h = len(x)
    n = h + 2
    x.insert(0, 1)
    x.append(256)

    a = x[0]
    b = x[1]

    l = ShannonEntropy(H, a, b)
    for i in range(1, n - 1):
        a = x[i] + 1
        b = x[i + 1]
        es = ShannonEntropy(H, a, b)
        l = l + es
    return l


def ShannonEntropy(H, a, b):
    a = round(a)
    b = round(b)

    h = H[a:b]

    x = h / sum(h)

    x = np.array(x)
    p = x[x > 0]

    # E = -∑(p(i)×log2(p(i)))
    S = -sum(p * [math.log(i, 2) for i in p])
    return S


# Unimodal benchmark functions
def F1(x):
    total = 0
    for i in range(len(x)):
        total += x[i] ** 2
    return total


def F2(x):
    sum = 0
    prod = 1
    for i in range(len(x)):
        sum += abs(x[i])
        prod *= abs(x[i])
    total = sum + prod
    return total


def F3(x):
    total = 0
    for i in range(len(x)):
        subTotal = 0
        for j in range(1, i + 1):
            subTotal += x[j]
        total += subTotal ** 2
    return total


def F4(x):
    ans = -1
    for i in range(len(x)):
        if abs(x[i]) > ans:
            ans = abs(x[i])
    return ans


def F5(x):
    total = 0
    for i in range(len(x) - 1):
        total += (100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2)
    return total


def F6(x):
    total = 0
    for i in range(len(x)):
        total += (x[i] + 0.5) ** 2
    return total


def F7(x):
    total = 0
    for i in range(1, len(x) + 1):
        total += i * x[i - 1] ** 4
    total += random.uniform(0, 1)
    return total


# Multi-modal benchmark functions
def F8(x):
    total = 0
    for i in range(len(x)):
        total += -x[i] * math.sin(math.sqrt(x[i]))
    return total


def F9(x):
    total = 0
    for i in range(len(x)):
        total += (x[i] ** 2 - (10 * math.cos(2 * math.pi * x[i])) + 10)
    return total


def F10(x):
    total1 = 0
    total2 = 0
    n = len(x)

    for i in range(n):
        total1 += x[i] ** 2
    total1 = total1 / n
    total1 = math.sqrt(total1)
    total1 = -0.2 * total1
    total1 = -20 * math.exp(total1)

    for i in range(n):
        total2 += math.cos(2 * math.pi * x[i])
    total2 = total2 / n
    total2 = math.exp(total2)

    return total1 - total2 + 20 + math.e


def F11(x):
    total = 0
    prod = 1

    for i in range(len(x)):
        total += x[i] ** 2
    total = total / 4000

    for i in range(1, len(x) + 1):
        prod *= math.cos(x[i - 1] / math.sqrt(i))
    return total - prod + 1


def F12(x):
    total = 0
    n = len(x)
    for i in range(n - 1):
        yi = 1 + ((x[i] + 1) / 4)
        yii = 1 + ((x[i + 1] + 1) / 4)
        total += ((yi - 1) ** 2) * (1 + 10 * math.sin(math.pi * yii))
    y1 = 1 + ((x[0] + 1) / 4)
    total += 10 * math.sin(math.pi * y1)
    yn = 1 + ((x[n - 1] + 1) / 4)
    total += (yn + 1) ** 2
    total = total * (math.pi / n)

    tot = 0
    a = 10
    k = 100
    m = 4
    for i in range(n):
        if x[i] > a:
            tot += k * ((x[i] - a) ** m)
        elif -a < x[i] < a:
            tot += 0
        else:
            k * ((-x[i] - a) ** m)

    return total + tot


def F13(x):
    total = 0
    total += math.sin(3 * math.pi * x[0])
    for i in range(len(x)):
        total += (x[i] - 1) ** 2 * (1 + math.sin(3 * math.pi * x[i] + 1) ** 2)
    total += (x[len(x) - 1] - 1) ** 2 * (1 + math.sin(2 * math.pi * x[len(x) - 1]) ** 2)
    total *= 0.1

    tot = 0
    a = 5
    k = 100
    m = 4
    for i in range(len(x)):
        if x[i] > a:
            tot += k * ((x[i] - a) ** m)
        elif -a < x[i] < a:
            tot += 0
        else:
            k * ((-x[i] - a) ** m)

    return total + tot
