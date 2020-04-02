################################################################################
#
#    compare.py - Analyse data of the spread of the COVID19 per Italian regions
#    usage: python3 compare.py
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
################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import wget
import ssl
import os

# get the data 
ssl._create_default_https_context = ssl._create_unverified_context
filename = wget.download('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')

# fit function
def line(x, a, b):
    return a*x+b

# read the csv data
w = pd.read_csv(filename)
os.remove(filename)
data = w.T.values.tolist()
d = { 'ill': 15,
      'inNeedForHospital': 8,
      'inNeedForICU': 9,
      'died': 14
      }

# make a plot
def doplot(data, which, relative = False, islog = False):
    regions = set(data[3])
    L = len(data[0])
    plt.figure(figsize=(12,7))
    x = []
    y = []
    # loop on regions
    for r in regions:
        threshold = 200
        # set a threshold for labelling curves
        lThreshold = threshold * 10
        l = r
        t = 0
        # loop on data
        for i in range(L):
            if data[3][i] == r:
                # count ills in the region
                ill = data[d[which]][i]
                if ill > threshold:
                    # if such a number is greater than the threshold, plot it
                    t += 1
                    x.append(t)
                    y.append(ill)
        if len(y) > 2 and y[-1] > threshold:
            ymax = y[-1]
            if relative:
                # if relative is True, plot data normalised to their maximum
                y = [i/ymax for i in y]
            if islog:
                # if log is True plot the logarithm of data
                y = [np.log(i) for i in y]
                p, cov = curve_fit(line, x, y)
                y = [y - p[1] for y in y]
                k = int(len(x)/2)
                l += ' [slope = {:.2f}]'.format(p[0])
                lThreshold = np.log(lThreshold) - p[1]
            if (y[-1] > lThreshold):
                # annotate regions with very high numbers
                plt.annotate(r, (x[-2], y[-1]))
            title = max(data[0])
            plt.plot(x, y, label = l)
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
    pngfile = 'compare'
    if relative:
        ylabel += ' (relative)'
        pngfile += '-relative'
    if islog:
        pngfile += '-log'
    plt.ylabel(ylabel)
    plt.xticks(rotation = 45)
    plt.title(title)
    plt.savefig(pngfile + '.png')
    plt.show()

doplot(data, 'ill')
doplot(data, 'ill', relative = True)
doplot(data, 'ill', islog = True)


