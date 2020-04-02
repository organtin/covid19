###########################################################################
#
#    covid19lib.py
#    A library for the analysis of the COVID19 outbreak
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
import numpy as np
import pandas as pd
from datetime import timedelta as td
import datetime as dt
from scipy.optimize import curve_fit
import wget
import ssl
import sys
import os
import re
from shutil import copyfile

# a logistic model
def flog(x, A, b, t0, C):
    return C + A/(1+np.exp(b*(x-t0)))

# the derivative of the logistic model
def dflog(x, A, b, t0):
    return -A*b*np.exp(b*(x-t0))/(1+np.exp(b*(x-t0)))**2

# the Gompterz function
def fgompertz(x, N0, b, c):
    return N0*np.exp(-b*np.exp(-c*x))

def dfgompertz(x, N0, b, c):
    return b*c*N0*np.exp(-b*np.exp(-c*x)-c*x)

# download data
def download(url, country, db, region):
    print('Getting data from {} for {}'.format(url.rstrip(), country))
    filename = wget.download(url)
    copyfile(filename, 'saved.csv')
    if db == 'Italy' and len(region) > 0:
        # remove useless columns from the input file: read it and write a new file
        fi = open(filename, 'r')
        ff = open('W' + filename, 'w')
        for line in fi:
            if re.search('^data.*', line) or re.search(region, line):
                # strip useless columns
                line = re.sub('ITA,[^,]+,[^,]+,[^,]+,[^,]+,', 'ITA,', line)
                ff.write(line)
        ff.close()
        fi.close()
        os.rename('W' + filename, filename)
    return filename

# The model function
def fun(x, a, b):
    return a*np.exp(x/b)

def getColumns(filename):
    w = pd.read_csv(filename)
    return w.columns.to_list()
    
def readData(filename, db, country, column):
    w = pd.read_csv(filename)
    columns = w.columns
    if db == 'Italy':
        data = w.T.values.tolist()
        head = [dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in data[0]]
        if re.search('\\+', column):
            ret = [0]*len(head)
            j = 0
            rcolumns = column.split('+')
            for j in range(len(data)):
                if columns[j] in rcolumns:
                    ret = [x + y for x, y in zip(ret, data[j])]
        else:
            tbr = columns.to_list().index(column)
            if len(country) > 0:
                tbr -= 3
            ret = data[tbr]
    else:
        data = w.values.tolist()
        head = [dt.datetime.strptime(x, '%m/%d/%y') for x in w.columns[4:]]
        i = 0;
        ret = [0]*len(head)
        # Select only data belonging to a given country as specified in column 1
        while i < len(data):
            if data[i][1] == country:
                for j in range(len(data[i]) - 4):
                    ret[j] += data[i][4 + j]
                print('{} {}'.format(data[i][0], data[i][1]))
            i += 1
    return ret, head

def dropLowStatistics(timeseries, min_val = 4):
    # Use only data with Nill > threshold. By default the threshold is 4
    i = 0
    if (len(sys.argv) > 2):
        min_val = int(sys.argv[2])
    print('Using only data with Nill > {}'.format(min_val))
    while (timeseries[i] < min_val):
        i += 1
    return i

def mkLabels(head, merge = 4, extend = False):
    xtlabels = []
    i = 0
    while i < len(head) - 1:
        xtlabels.append(str(head[i + 1]).split(' ')[0][5:])
        i += merge
    if extend:
        lastlabel = head[len(head) - 1]
        i = 0
        # add more ticks
        while i < len(head):
            lastlabel += td(days = merge)
            xtlabels.append(str(lastlabel).split(' ')[0][5:])
            i += merge  
    return xtlabels

def computeDifferences(head, timeseries, merge = 4):
    # compute new infected, then aggregate data by two consecutive days
    Ntemp = []
    NnewComputed = []
    for i in range(len(head) - 1):
        Ntemp.append(timeseries[i + 1] - timeseries[i])
    i = 0
    while (i < len(Ntemp) - 1):
        Nn = 0
        for k in range(merge):
            if i + k < len(Ntemp):
                dNn = Ntemp[i + k]
            else:
                dNn = 0
            Nn += dNn
        NnewComputed.append(Nn)
        i += merge
    xx = range(len(NnewComputed))
    # normalise to their maximum
    M = max(NnewComputed)
    NnewComputed = [x/M for x in NnewComputed]
    return xx, NnewComputed

def getCleanData(filename, db, column, region = '', drop = 0):
    Nill, head = readData(filename, db, region, column)
    i = dropLowStatistics(Nill)
    Nill = Nill[i:]
    head = head[i:]
    if drop > 0:
        Nill = Nill[:-drop]
        head = head[:-drop]
    return Nill, head

# plot the result
def barplot(x, head, y, dy, p, tpeak, country, merge = 4, title = 'totale_casi', fun = dflog):
    title += ' (differences)'
    plt.figure(figsize=(12,7))
    plt.bar(x, y, yerr = dy, label = 'aggregating {} days'.format(merge))
    daysLeft = merge - ((len(y) - 1) % merge)
    plt.annotate('Last data taken on ' + str(head[-1]), (0.1, 0.9), xycoords = 'axes fraction')
    if daysLeft > 0:
        plt.annotate('Last bin missing data from {} days\n(not fitted)'.format(daysLeft),
                     (0.1,0.8), xycoords = 'axes fraction')
    xx2 = range(2*max(x))
    tpeaks = str(head[0]+ td(days = tpeak)).split(' ')[0]
    plt.plot(xx2, fun(xx2, p[0], p[1], p[2]), '-',
             label = 'dL/dt Fit ($t_0={}$)'.format(tpeaks), color='orange')
    plt.legend(loc = 'upper right')
    xtlabels = mkLabels(head)
    plt.xticks(xx2, xtlabels, rotation = 90)
    plt.xlabel('t [d]')
    plt.ylabel(title)
    plt.title(country + '\n' + title)
    plt.savefig('derivative-of-logistics.png')
    plt.show()

def computeTimes(p, cov, fun, merge):
    if fun == flog:
        t0 = p[2] * merge
        dt0 = np.sqrt(cov[2][2])
        tr = 1/p[1]
        dTr = np.sqrt(cov[1][1])/(p[1]**2)
    else:
        t0 = -np.log(np.log(2)/p[1])/p[2]
        s1 = cov[1][1]/(p[1]*p[2])**2
        s2 = cov[2][2]*(t0/p[2])**2
        dt0 = np.sqrt(s1+s2)
    return t0, dt0
        
def doplot(x, y, head, p, cov, country, title = 'totale_casi', fun = flog, merge = 4):
    t0, dt0 = computeTimes(p, cov, fun, merge)
    tr = 2*t0
    dTr = 2*dt0
    M = max(y)
    ynorm = [y/M for y in y]
    dyNorm = [np.sqrt(y)/M for y in y]
    pL, cov = curve_fit(fun, x, ynorm, sigma=dyNorm, maxfev=10000)
    t1, dt1 = computeTimes(pL, cov, fun, 1)
    xr2 = range(2*int(np.ceil(t0)))
    print('L(t) fit parameters: ')
    fd = open('L.results', 'a+')
    line = str(head[-1])
    for i in range(len(pL)):
        line += ' ' + str(pL[i])
    fd.write(line + '\n')
    plt.figure(figsize=(12,7))
    if fun == flog:
        title += ' Logistic model'
    else:
        title += ' Gompertz'
    plt.title('Evolution of COVID19 spread with time\n' + title)
    plt.plot(x, y, 'o', label = '[{}] Data up to {}'.format(country, head[-1]))
    tpeak = t0
    if fun == flog:
        plt.plot(xr2, M*fun(xr2, pL[0], pL[1], pL[2], pL[3]), '-')
    else:
        plt.plot(xr2, M*fun(xr2, pL[0], pL[1], pL[2]), '-')
    plt.axvspan(t1-dt1 , t1+dt1, alpha=0.5, color='orange')
    plt.axvline(x=t1, color='orange', label = 'Peak estimated from L(t)')
    plt.axvspan(t0-dt0 , t0+dt0, alpha=0.5, color='red')
    plt.axvline(x=t0, color='red', label='Peak estimated using dL/dt')
    dumpResult(head[-1], pL, merge)
    plt.legend()
    plt.xlabel('t [d]')
    plt.ylabel('N')
    xtlabels = mkLabels(head, merge = 1, extend = True)
    plt.xticks(xr2, xtlabels, rotation = 45)
    plt.savefig('logisticfit.png')
    plt.show()

# fit
def doFit(head, x, y, dy, merge, fun = dflog):
    # fit the data with the derivative of the logistic curve
    lastbin = len(x)
    daysLeft = merge - ((len(y) - 1) % merge)
    if daysLeft > 0:
        lastbin = -1
    p, cov = curve_fit(fun, x[:lastbin], y[:lastbin],
                       sigma = dy[:lastbin], maxfev=10000)
    print('=============== {} ===================='.format(fun))
    print('dL/dt fit parameters: ')
    tpeak = dumpResult(head[-1], p, merge, fun = fun)
    return p, cov, tpeak

def dumpResult(lastDay, p, merge, fun = flog):
    print(p)
    if fun == dflog:
        tpeak = p[2] * merge
    else:
        tpeak = -np.log(np.log(2)/p[1])/p[2]
    tr = 2*tpeak
    print('Date of the peak   : {}'.format(lastDay + td(days = tpeak)))
    fd = open('dLdt.results', 'a+')
    line = str(lastDay)
    for i in range(len(p)):
        line += ' ' + str(p[i])
    fd.write(line + '\n')
    return tpeak

# start of the analysis
def analyse(url, country, db, region, merge = 4, drop = 0, column='totale_casi'):
    filename   = download(url, country, db, region)
    Nill, head = getCleanData(filename, db, column, region, drop)
    dNill = np.sqrt(Nill)
    M = max(Nill)
    xx, NnewComputed = computeDifferences(head, Nill)
    dNr = [x/M for x in NnewComputed]
    p, cov, tpeak = doFit(head, xx, NnewComputed, dNr, merge)
    barplot(xx, head, NnewComputed, dNr, p, tpeak, country, title = column)
    doplot(range(len(Nill)), Nill, head, p, cov, country, title = column, merge = merge)
    pg, cov, tpeak = doFit(head, xx, NnewComputed, dNr, merge, fun = dfgompertz)
    barplot(xx, head, NnewComputed, dNr, pg, tpeak, country, title = column, fun = dfgompertz)
    doplot(range(len(Nill)), Nill, head, pg, cov, country, title = column,
           fun = fgompertz, merge = merge)
    return filename

def plotRatio(numerator, denominator, db = 'Italy', region = '', fname = ''):
    num, t   = readData(fname, db, region, numerator)
    den, t = readData(fname, db, region, denominator)
    while den[0] == 0:
        den.pop(0)
        num.pop(0)
        t.pop(0)
    rdi = [x/y for x, y in zip(num, den)]
    drdi = [np.sqrt(1/x + 1/y)*x/y for x, y in zip(num, den)]
    plt.figure(figsize=(12,7))
    plt.errorbar(t, rdi, yerr = drdi, fmt = 'o')
    plt.title(numerator + '/' + denominator)
    plt.xticks(rotation = 45)
    pngfile = numerator + denominator
    plt.ylim(0,1)
    plt.savefig(pngfile + '.png')
    plt.show()

