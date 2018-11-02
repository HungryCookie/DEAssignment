import matplotlib.pyplot as plt
import numpy as np


def funct(x, y):
    return - y - x


# Implementation of Euler's Method of approximation
def euler(x0, y0, x, h):
    ya, xa = [y0], [x0]
    while x0 < x:
        y0 += h * funct(x0, y0)
        x0 += h
        xa.append(x0)
        ya.append(y0)
    return xa, ya


# Implementation of Improved Euler Method of approximation
def improvedEuler(x0, y0, x, h):
    ya, xa = [y0], [x0]
    while x0 < x:
        k1 = funct(x0, y0)
        k2 = funct(x0 + h, y0 + h * k1)
        y0 += h/2 * (k1 + k2)
        x0 += h
        xa.append(x0)
        ya.append(y0)

    return xa, ya


# Implementation of Runge-Kutta Method of approximation
def rk(x0, y0, x, h):
    ya, xa = [y0], [x0]
    while x0 < x:
        k1 = funct(x0, y0)
        k2 = funct(x0 + h/2, y0 + h/2 * k1)
        k3 = funct(x0 + h/2, y0 + h/2 * k2)
        k4 = funct(x0 + h, y0 + h * k3)
        y0 += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        x0 += h
        xa.append(x0)
        ya.append(y0)
    return xa, ya


# Exact solution to the given equation
def exact(x0, y0, x, h):
    ya, xa = [y0], [x0]
    while x0 < x:
        x0 += h
        y0 = 1 - x0
        xa.append(x0)
        ya.append(y0)
    return xa, ya


x0, y0, x, h = 0, 1, 10, 0.001  # Initial Values

x1, y1 = euler(x0, y0, x, h)
x2, y2 = improvedEuler(x0, y0, x, h)
x3, y3 = rk(x0, y0, x, h)
x4, y4 = exact(x0, y0, x, h)

# Calculating the error of approximation
error1, error2, error3 = [], [], []
for i in range(0, len(y4)):
    error1.append(np.abs(y4[i] - y1[i]))
    error2.append(np.abs(y4[i]- y2[i]))
    error3.append(np.abs(y4[i] - y3[i]))

# Plotting approximation graphs
plt.figure(1)
plt.plot(x1, y1, label = 'Euler')
plt.plot(x2, y2, label = 'Improved Euler')
plt.plot(x3, y3, label = 'Runge-Kutta')
plt.plot(x4, y4, label = 'Exact')
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Methods of Approximation")
plt.legend()

# Plotting graphs of approximation error
plt.figure(2)
plt.plot(x4, error1, label = 'Euler')
plt.plot(x4, error2, label = 'Improved Euler')
plt.plot(x4, error3, label = 'Runge-Kutta')
plt.ylim(0, 0.000000001)
plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Errors")
plt.legend()

plt.show()
