###########################################################################                                     
#                                                                                                               
#    trend.py - Analyse data of the spread of the COVID19                                                       
#    usage: python3 trend.py [name of the country] [minimum number of ills]                                     
#    Copyright (C) 2020 giovanni.organtini@uniroma1.it                                                          
#                                                                                                               
#    This program is free software: you can redistribute it and/or modify                                       
#    it under the terms of the GNU General Public License as published by                                       
#    the Free Software Foundation, either version 3 of the License, or                                          
#    (at your option) any later version.                                                                        
#                                                                                                               
#    This program is distributed in the hope that it will be useful,                                            
#    but WITHOUT ANY WARRANTY; without even the implied warranty of                                             
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                                              
#    GNU General Public License for more details.                                                               
#                                                                                                               
#    You should have received a copy of the GNU General Public License                                          
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.                                   
#                                                                                                             
###########################################################################  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

relative = False
islog = False
which = 'ill'

def line(x, a, b):
    return a*x+b

w = pd.read_csv('saved.csv')
data = w.T.values.tolist()
L = len(data[0])
print(L)
regions = set(data[3])
x = []
y = []
d = { 'ill': 14,
      'inNeedForHospital': 8,
      'inNeedForICU': 9,
      'died': 13
      }
l = ''
plt.figure(figsize=(12,7))
for r in regions:
    t = 0
    for i in range(L):
        if data[3][i] == r:
            ill = data[d[which]][i]
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
            y = [y - p[1] for y in y]
            k = int(len(x)/2)
            l = r + ' [slope = {:.2f}]'.format(p[0])
        title = max(data[0])
        if len(l) > 0:
            plt.plot(x, y, label = l)
        else:
            plt.plot(x,y)
    x.clear()
    y.clear()
if len(l) > 0:
    plt.legend()
plt.xlabel('time from t$_0$ where t$_0$ is the time at which N$_{inf}>100$ [d]')
ylabel = 'N_{infected}'
if which == 'inNeedForHospital':
    ylabel = 'N_{hospital}'
elif which == 'inNeedForICU':
    ylabel = 'N_{ICU}'
elif which == 'died':
    ylabel = 'N_{died}'
if islog:
    ylabel = '\log{' + ylabel + '}'
ylabel = '$' + ylabel + '$'
if relative:
    ylabel += ' (relative)'
plt.ylabel(ylabel)
plt.xticks(rotation = 45)
plt.title(title)
plt.show()
        

