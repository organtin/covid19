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
import math

drop = 0

# You can download the data from the following URL. Data are expected to be organised as
# in the given CSV file. 

ssl._create_default_https_context = ssl._create_unverified_context
urls = {
#    'World' : 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv',
    'World' : 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
    'Italy' : 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv',
    'Regional': 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'
    }

url = urls['Italy']
db = 'Italy'
country = 'Italy'
region = ''

# help function
def covidhelp():
    print('Usage: {} [source] [min]'.format(sys.argv[0]))
    print('       [source] is optional and can be either "Italy[/<region>]" or "World/<country>"')
    print('       where <country> is the name of the country in which you are')
    print('       interested in (e.g. "World/China" or "World/Norway").')
    print('       The deafult source is Italy. Using Italy/Lazio you use only data for Lazio.')
    print('       [min] identifies the starting point of the plot. Data are considered only')
    print('       starting from the date at which the number of infected begins to be ')
    print('       higher than [min] (default to 4).')


# a logistic model
def flog(x, A, b, t0, C):
    return C + A/(1+np.exp(b*(x-t0)))

# the derivative of the logistic model
def dflog(x, A, b, t0):
    return -A*b*np.exp(b*(x-t0))/(1+np.exp(b*(x-t0)))**2

# get arguments, if any
if (len(sys.argv) > 1):
    if not re.match('.*/', sys.argv[1]):
        sys.argv[1] += "/"
    (db, region) = sys.argv[1].split("/")
    if db == 'Italy' and len(region) > 0:
        url = urls['Regional']
        country = region
    elif db == 'World': 
        url = urls['World']
        country = sys.argv[1].split("/")[1]
    elif not (db == 'Italy' and len(region) == 0):
        covidhelp()
        exit(0)

# download data
print('Getting data from {} for {}'.format(url.rstrip(), country))
filename = wget.download(url)
copyfile(filename, 'saved.csv')
if db == 'Italy' and len(region) > 0:
    # remove useless columns from the input file: read it and write a new file
    fi = open(filename, 'r')
    ff = open('W' + filename, 'w')
    for line in fi:
        if re.search(region, line):
            # strip useless columns
            line = re.sub('ITA,[^,]+,[^,]+,[^,]+,[^,]+,', 'ITA,', line)
            ff.write(line)
    ff.close()
    fi.close()
    os.rename('W' + filename, filename)

# The model function
def fun(x, a, b):
    return a*np.exp(x/b)

w = pd.read_csv(filename)
os.remove(filename)
if db == 'Italy':
    data = w.T.values.tolist()
    head = [dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') for x in data[0]]
    Nill = data[11]
    Nnewills = data[7]
    Ndeaths = data[10]
    Nrecovered = data[9]
    Nsymptoms = data[2]
    Nisolated = data[5]
    Nhospital = data[4]
    Ncritical = data[3]
    Ntested = data[12]
else:
    w = w[w['Country/Region'] == country]
    data = w.values.tolist()
    head = [dt.datetime.strptime(x, '%m/%d/%y') for x in w.columns[4:]]    
    Nill = data[0][4:]
    for i in range(1, w.shape[0]):
        Nill = [sum(x) for x in zip(Nill, data[i][4:])]
        print(Nill[-1])
    print(Nill)

# Use only data with Nill > threshold. By default the threshold is 4
i = 0
min_val = 4
if (len(sys.argv) > 2):
    min_val = int(sys.argv[2])
print('Using only data with Nill > {}'.format(min_val))
while (Nill[i] < min_val):
    i += 1

Nill = Nill[i:]
head  = head[i:]
if drop > 0:
    Nill = Nill[:-drop]
    head = head[:-drop]
if (db != 'World'):
    Ndeaths = Ndeaths[i:]
    Nnewills = Nnewills[i:]
    Nrecovered = Nrecovered[i:]
    Nhospital = Nhospital[i:]
    Ncritical = Ncritical[i:]
    Nsymptoms = Nsymptoms[i:]
    Nisolated = Nisolated[i:]
    Ntested = Ntested[i:]
    # make the Longo plot (cumulating patients in isolation, in hospital and in ICU)
    NcumIsol = []
    NcumSymp = []
    NcumCrit = []
    NcumErr  = []
    for i in range(len(head) - 1):
        N1 = Nisolated[i]
        N2 = Nisolated[i + 1]
        NcumIsol.append(N2-N1)
        N1 = Nsymptoms[i]
        N2 = Nsymptoms[i + 1]
        NcumSymp.append(N2-N1)
        N1 = Ncritical[i]
        N2 = Ncritical[i + 1]
        NcumCrit.append(N2-N1)
        NcumErr = np.sqrt(NcumIsol[-1] + NcumSymp[-1] + NcumCrit[-1])
    plt.figure(figsize=(12,7))
    pbIsol = plt.bar(head[:-1], NcumIsol)
    bottom = NcumIsol
    for i in range(len(bottom)):
        if bottom[i] < 0:
            bottom[i] = 0
    pbSymp = plt.bar(head[:-1], NcumSymp, bottom = bottom)
    nOthers = [x + y for x, y in zip(NcumIsol, NcumSymp)]
    bottom = nOthers
    for i in range(len(bottom)):
        if bottom[i] < 0:
            bottom[i] = 0
    pbCrit = plt.bar(head[:-1], NcumCrit, bottom = bottom, yerr = NcumErr)
    plt.xlabel('t [d]')
    plt.ylabel('$N_{isolated} + N_{with symptoms} + N_{crit}$')
    plt.title(country)
    plt.xticks(rotation=45)
    plt.legend([pbIsol[0], pbCrit[0], pbSymp[0]], ['in isolation', 'in critical conditions', 'with symptoms'])
    plt.savefig('longoplot.png')
    plt.plot()

# compute new infected, then aggregate data by two consecutive days
Ntemp = []
NnewComputed = []
for i in range(len(head) - 1):
    Ntemp.append(Nill[i + 1] - Nill[i])
i = 0
merge = 4
underestimated = 0
#while (i < len(Ntemp) - 1):
while (i < len(Ntemp)):
    Nn = 0
    for k in range(merge):
        if i + k < len(Ntemp):
            dNn = Ntemp[i + k]
        else:
            dNn = 0
            underestimated += 1
        Nn += dNn
    NnewComputed.append(Nn)
    i += merge
xx = range(len(NnewComputed))
# compute the unicertainties of the new infected
dNr = np.sqrt(NnewComputed)
# normalise to their maximum
M = max(NnewComputed)
NnewComputed = [x/M for x in NnewComputed]
dNr = [x/M for x in dNr]
# fit the data with the derivative of the logistic curve
lastbin = len(xx)
if underestimated > 0:
    lastbin = -1
p, cov = curve_fit(dflog, xx[:lastbin], NnewComputed[:lastbin], sigma = dNr[:lastbin], maxfev=10000)
# plot the result
print('t0 = {:.0f} from now'.format(len(xx) - p[2]))
plt.figure(figsize=(12,7))
plt.bar(xx, NnewComputed, yerr = dNr,
        label = 'New infected - aggregating {} days'.format(merge))
if underestimated > 0:
    plt.annotate('Last bin missing data from {} days\n(not fitted)'.format(underestimated), (0.1,0.9))
xx2 = range(2*max(xx))
tpeak = str(head[0] + td(days = merge*p[2]))
tpeak = tpeak.split(' ')[0]
plt.plot(xx2, dflog(xx2, p[0], p[1], p[2]), '-',
         label = 'dL/dt Fit ($t_0={}$)'.format(tpeak), color='orange')
plt.legend()
i = 1
xtlabels = []
while i < len(head) - 1:
    xtlabels.append(str(head[i + 1]).split(' ')[0][5:])
    i += merge
lastlabel = head[len(head) - 1]
i = 0
# add more ticks
while i < len(head):
    lastlabel += td(days = merge)
    xtlabels.append(str(lastlabel).split(' ')[0][5:])
    i += merge    
plt.xticks(xx2, xtlabels, rotation = 90)
plt.xlabel('t [d]')
plt.ylabel('$N_{new\\,infected}$ (computed)')
plt.title(country + '\nFit with the derivative of logistic curve')
plt.savefig('derivative-of-logistic.png')
plt.show()

# Compute the logarithm of the data
lNill = [np.log(x) for x in Nill]

# A function to perform a fit
def fit(y, n, m=-1):
    if m < 0:
        m = len(y)-1
    print('Fitting points from {} to {}'.format(n, m))
    Nr = y[n:m]
    xr = np.arange(n,m)
    dNr = [np.sqrt(x) for x in Nr]
    p, cov = curve_fit(fun, xr, Nr, sigma=dNr, maxfev=10000)
    s0 = np.sqrt(cov[0][0])
    s1 = np.sqrt(cov[1][1])
    return p[0], p[1], s0, s1

# Fit the last data with the model
A, tau, dA, dtau = fit(Nill, len(Nill)-9)
print('Fit done: tau = {} +- {}'.format(tau, dtau))
print('            A = {} +- {}'.format(A, dA))

# A function to make a nice plot
def myplotfit(plt, x, y, p0, p1, s0, s1, legendPosition=0, legend=None, title=None, fmt = 'o'):
    plt.xticks(rotation=45)
    plt.plot(x, np.log(y), fmt)
    xr = range(len(x))
    if legend != None:
        ymin, ymax = plt.ylim()
        plt.annotate(legend, (0.1, legendPosition), xycoords='axes fraction')
    s1 = p0*np.exp(xr/p1)*xr/p1**2*s1
    s0 = np.exp(xr/p1)*s0
    s  = np.sqrt(s1**2+s0**2)
    f0 = fun(xr, p0, p1)
    f1 = fun(xr, p0, p1)+s
    f2 = fun(xr, p0, p1)-s
    plt.plot(x, np.log(f0))
    if title != None:
        plt.title(title)
    plt.fill_between(x, np.log(f1), np.log(f2), alpha=0.25)

# Fit a portion of the data iteratively
intervalAmplitude = 6
xmax = len(head) - 1
xmin = xmax - intervalAmplitude
plt.figure(figsize=(12,7))
plt.title(country)
legendPos = 0.9
taudata = [0]*len(head)
while xmin >= 0:
    # Fit the given portion of data
    A, tau, dA, dtau = fit(Nill, xmin, xmax)
    taudata[xmin] = tau
    print('Fit done: tau = {} +- {} ({})'.format(tau, dtau, dtau/tau))
    print('            A = {} +- {} ({})'.format(A, dA, dA/A))
    # Add the model to the plot
    label = 'Fit from ' + str(xmin) + ' to ' + str(xmax) + ': $\\tau = {:.2f}$ d'.format(tau) 
    myplotfit(plt, head, Nill, A, tau, dA, dtau, legendPos, label)
    xmax -= intervalAmplitude
    xmin = xmax - intervalAmplitude
    legendPos -= 0.1

# Show the plot with the various models superimposed
plt.savefig('multimodel.png')
plt.show()

# Make a plot of the evolution of the characteristic time
plt.figure(figsize=(12,7))
plt.plot(head, taudata, 'o')
plt.ylim(bottom = 0.1)
plt.xticks(rotation=45)
plt.title(country)
plt.ylabel('Characteristic time [d]')
plt.show()

# make a plot of the derivative of log(N(t))
deriv = []
i = 0
for i in range(len(lNill) - 1):
    deriv.append(1/(lNill[i + 1]-lNill[i]))
    i += 1
plt.figure(figsize=(12,7))
plt.title('Evolution of the characteristic time of coronavirus spread\n{}'.format(country))
plt.plot(head[:-1], deriv, '-o')
plt.ylabel('$\\frac{d\\tau}{dt}$')
plt.xticks(rotation=45)
plt.savefig('dtaudt.png')
plt.show()

# normalisation function
def normalise(v):
    return [x/max(v) for x in v]

# the Gompterz function dev +
def fgompertz(x, N0, b, c):
    return N0*np.exp(-b*np.exp(-c*x))

def dfgompertz(x, N0, b, c):
    return b*c*N0*np.exp(-b*np.exp(-c*x)-c*x)

# normalise data
NillNorm = normalise(Nill)

# fit and plot normalised data
xr = range(len(NillNorm))
pg, cov = curve_fit(fgompertz, xr, NillNorm, sigma=np.sqrt(NillNorm), maxfev=10000)
print('==== Gompertz +')
print(pg)
# p[0] = max level
# p[1] = displacemente along t
# p[2] = slope
thwp = -np.log(np.log(2)/pg[1])/pg[2]
print('t_hwp = {}'.format(thwp))
print('t_max = {}'.format(np.log(pg[1])/pg[2]))
xr2 = range(150)
plt.plot(xr2, fgompertz(xr2, pg[0], pg[1], pg[2]), '-',
         label = 'Gompertz fit: $t(1/2) = {:.0f}$ d'.format(thwp))
plt.plot(xr, NillNorm, 'o', label = '[{}] Data up to {}'.format(country, head[-1]))
plt.xlabel('t from case {} [d]'.format(min_val))
plt.ylabel('$N_{infected}$')
plt.legend()
plt.show()
pgd, cov = curve_fit(dfgompertz, xx[:lastbin], NnewComputed[:lastbin],
                     sigma=np.sqrt(NnewComputed[:lastbin]), maxfev=10000)
print(pgd)
nthwp = -np.log(np.log(2)/pgd[1])/pgd[2]
t_halfpeak = str(head[0] + td(days = merge * (nthwp - 1))).split(' ')[0]
print('t(1/2) = {}'.format(t_halfpeak))
xr2 = range(2*len(xx))
plt.figure(figsize=(12,7))
plt.plot(xr2, dfgompertz(xr2, pgd[0], pgd[1], pgd[2]), '-',
         label = 'Gompertz derivative - t(1/2): {}'.format(t_halfpeak))
plt.plot(xx, NnewComputed, 'o', label = '[{}] Data up to {}'.format(country, head[-1]))
if underestimated > 0:
    plt.annotate('Last bin missing data from {} days\n(not fitted)'.format(underestimated), (0.1,0.9))
plt.xticks(xx2, xtlabels, rotation = 90)
plt.xlabel('t [d]')
plt.ylabel('$N_{new\\,infected}$ (computed)')
plt.legend()
plt.savefig('gompertz-derivative.png')
plt.show()
print('==== Gompertz -')
p, cov = curve_fit(flog, xr, NillNorm, sigma=np.sqrt(NillNorm), maxfev=10000)
print(p)
t0 = p[2]
dt0 = np.sqrt(cov[2][2])
A = 1/p[3]
dA = np.sqrt(cov[3][3])/(p[3]**2)
tr = 1/p[1]
dTr = np.sqrt(cov[1][1])/(p[1]**2)
xr2 = range(2*int(np.ceil(t0)))
ttp = t0+3*tr-max(xr)
plt.figure(figsize=(12,7))
plt.title('Evolution of COVID19 spread with time (tentative)\nLogistic model')
dflogNorm = p[3]/dflog(p[2], p[0], p[1], p[2])
ampli = dflogNorm*dflog(t0, p[0], p[1], p[2])
plt.plot(xr2, dflogNorm*dflog(xr2, p[0], p[1], p[2]), '-', label = 'Logistic derivative (amplified)')
plt.plot(xr2, fgompertz(xr2, pg[0], pg[1], pg[2]), '-', label = 'Gompertz')
if pg[0] > ampli:
    plt.ylim(0, ampli*1.2)
plt.plot(xr, NillNorm, 'o', label = '[{}] Data up to {}'.format(country, head[-1]))
tpeak = t0 - max(xr)
plt.plot(xr2, flog(xr2, p[0], p[1], p[2], p[3]), '-',         
         label = 'Logistic model\nTime to plateau: {:.0f} d\nTime to peak: {:.0f} d'.format(ttp, tpeak))
print('Date of the peak   : {}'.format(head[-1] + td(days = tpeak)))
print('Date of the plateau: {}'.format(head[-1] + td(days = ttp)))
plt.legend()
plt.xlabel('t [d]')
plt.ylabel('N')
print('Time t0        : {:.2f} +- {:.2f}'.format(t0, dt0))
print('Current level  : {:.2f} +- {:.2f}'.format(A, dA))
print('Rise time      : {:.2f} +- {:.2f}'.format(tr, dTr))
print('Current time   : {}'.format(max(xr)))
print('Time to plateau: {:.0f} (estimate)'.format(ttp))
plt.savefig('logisticfit.png')
plt.show()

# plot normalised data about other categories
if db == 'Italy':
    Npop = 60e6

    derDeath = []
    derReco = []
    derHosp = []
    derCrit = []
    for i in range(len(Nill) - 1):
        derDeath.append(Ndeaths[i + 1] - Ndeaths[i])
        derReco.append(Nrecovered[i + 1] - Nrecovered[i])
        derHosp.append(Nhospital[i + 1] - Nhospital[i])
        derCrit.append(Ncritical[i + 1] - Ncritical[i])
    xder = xr[:-1]
    plt.figure(figsize=(12,7))
    plt.title('Derivative of categories\n{}'.format(country))
    plt.plot(xder, derDeath, '-o', label = 'Deaths')
    plt.plot(xder, derReco, '-o', label = 'Recovered')
    plt.plot(xder, derHosp, '-o', label = 'In need of hospital')
    plt.plot(xder, derCrit, '-o', label = 'in need for ICU')
    plt.xlabel('t [d]')
    plt.ylabel('dN/dt [d$^{-1}$]')
    plt.legend()
    plt.savefig('derivatives.png')
    plt.show()
    
    NdeathNorm = normalise(Ndeaths)
    NrecoNorm = normalise(Nrecovered)
    NhospNorm = normalise(Nhospital)
    NcritNorm = normalise(Ncritical)

    rInfected = max(Nill)/Npop*1000
    rDeaths = max(Ndeaths)/Npop*1000

    rDeaths2Ill = max(Ndeaths)/max(Nill)*100
    rReco2Ill = max(Nrecovered)/max(Nill)*100
    rHosp2Ill = max(Nhospital)/max(Nill)*100
    rCrit2Ill = max(Ncritical)/max(Nill)*100
    
    plt.figure(figsize=(12,7))
    plt.title('Evolution of COVID19 spread with time')
    lgdI, = plt.plot(xr, NillNorm, '-o',
                     label='Infected ({}) {:.2f} permille of population'.format(max(Nill), rInfected))
    label = 'Death ({}) {:.2f} permille of population/{:.2f}% of infected'.format(max(Ndeaths),
                                                                                  rDeaths, rDeaths2Ill)
    plt.plot(xr, NdeathNorm, '-o', markerfacecolor='w', label=label)
    label = 'Recovered ({}) {:.2f}% of infected'.format(max(Nrecovered), rReco2Ill)
    plt.plot(xr, NrecoNorm, '-o', markerfacecolor='w', label=label)
    label = 'Inpatient ({}) {:.2f}% of infected'.format(max(Nhospital), rHosp2Ill)
    plt.plot(xr, NhospNorm, '-o', markerfacecolor='w', label=label)
    label = 'Critical ({}) {:.2f}% of infected'.format(max(Ncritical), rCrit2Ill)
    plt.plot(xr, NcritNorm, '-o', markerfacecolor='w', label=label)
    plt.annotate('[{}] Data up to {}'.format(country, head[-1]), (0.1, 0.6), xycoords='axes fraction')
    plt.xlabel('t [d]')
    plt.ylabel('N/N$_{max}$')
    plt.legend()
    plt.savefig('normalised.png')
    plt.show()

    mortality = []
    em = []
    inNeedForHospital = []
    eh = []
    inNeedForIntensiveCareUnit = []
    eicu = []
    for i in range(len(Nill)):
        N2 = Nill[i]
        N1 = Ndeaths[i]
        e1 = np.sqrt(N1)/N2
        e2 = N1*np.sqrt(N2)/N2**2
        e = np.sqrt(e1**2 + e2**2)
        mortality.append(N1/N2)
        em.append(e)
        N1 = Nhospital[i]
        e1 = np.sqrt(N1)/N2
        e2 = N1*np.sqrt(N2)/N2**2
        e = np.sqrt(e1**2 + e2**2)
        inNeedForHospital.append(N1/N2)
        eh.append(e)
        N1 = Ncritical[i]
        e1 = np.sqrt(N1)/N2
        e2 = N1*np.sqrt(N2)/N2**2
        e = np.sqrt(e1**2 + e2**2)
        eicu.append(e)
        inNeedForIntensiveCareUnit.append(N1/N2)

    plt.figure(figsize=(12,7))
    plt.title('COVID19: evolution of the consequences of infection')
    plt.errorbar(xr, mortality, label='mortality', yerr=em, fmt='-o')
    plt.errorbar(xr, inNeedForHospital, yerr=eh, fmt='-o', label='in need for hospital')
    plt.errorbar(xr, inNeedForIntensiveCareUnit, yerr=eicu, fmt='-o', label='in need for intensive care unit')
    plt.xlabel('t [d]')
    plt.ylabel('N/N$_{infected}$')
    plt.legend()
    plt.savefig('consequences.png')
    plt.show()

    infectedOverTested = [x/y for x, y in zip(Nill, Ntested)]
    plt.plot(xr, infectedOverTested)
    plt.xlabel('t [d]')
    plt.ylabel('Infected/Tested')
    plt.show()
