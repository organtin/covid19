import matplotlib.pyplot as plt
import pandas as pd

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
    plt.plot(x, y, label = r)
    if len(y) > 0 and y[-1] > 1000:
        plt.annotate(r, (x[-3], y[-2]))
    x.clear()
    y.clear()
plt.legend()
plt.xlabel('time from t$_0$ where t$_0$ is the time at which N$_{inf}>100$ [d]')
plt.ylabel('N$_{inf}$')
plt.xticks(rotation = 45)
plt.show()
        

