###########################################################################
#
#    trend3.py - Analyse data of the spread of the COVID19
#    usage: python3 trend3.py [name of the country] [minimum number of ills]
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
import covid19lib as c19
import sys
import ssl
import os
import re

ssl._create_default_https_context = ssl._create_unverified_context
urls = {
    'World' : 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
    'Italy' : 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv',
    'Regional': 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'
    }

url = urls['Italy']
db = 'Italy'
country = 'Italy'
region = ''

print(sys.argv)
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


filename = c19.download(url, country, db, region)

if db == 'Italy':
    c19.plotRatio('deceduti', 'totale_casi', fname = filename)
    c19.plotRatio('ricoverati_con_sintomi+terapia_intensiva+isolamento_domiciliare',
                  'totale_casi', fname = filename)
    c19.plotRatio('totale_ospedalizzati', 'totale_casi', fname = filename)
    c19.plotRatio('terapia_intensiva', 'totale_casi', fname = filename)
    c19.plotRatio('totale_casi', 'tamponi', fname = filename)

i = 0
#os.remove('dLdt.results')
#os.remove('L.results')
#for i in range(15):
#    c19.analyse(url, country, db, region, column = 'totale_casi', drop = i)

#cols = c19.getColumns(filename)
#for i in range(2, 11):
#    c19.analyse(url, country, db, region, column = cols[i])

c19.analyse(url, country, db, region, column = 'deceduti')    

os.remove(filename)
