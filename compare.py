import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

relative = True
islog = True

def line(x, a, b):
    return a*x+b

w = pd.read_csv('saved.csv')
data = w.T.values.tolist()
L = len(data[0])
print(L)
regions = set(data[3])
x = []
y = []
plt.figure(figsize=(12,7))
for r in regions:
    t = 0
    for i in range(L):
        if data[3][i] == r:
            ill = data[14][i]
            if ill > 100:
                t += 1
                x.append(t)
                y.append(ill)
    if len(y) > 0 and y[-1] > 1000:
        plt.annotate(r, (x[-3], y[-2]))
        ymax = y[-1]
        if relative:
            y = [i/ymax for i in y]
        if islog:
            y = [np.log(i) for i in y]
            p, cov = curve_fit(line, x, y)
        plt.plot(x, y, label = r)
        k = int(len(x)/2)
        plt.annotate('slope = {:.2f}'.format(p[0]), (x[k], y[k]))
    x.clear()
    y.clear()
plt.legend()
plt.xlabel('time from t$_0$ where t$_0$ is the time at which N$_{inf}>100$ [d]')
ylabel = 'N$_{inf}$'
if islog:
    ylabel = '$\log{N_{inf}}$'
if relative:
    ylabel += ' (relative)'
plt.ylabel(ylabel)
plt.xticks(rotation = 45)
plt.show()
        

