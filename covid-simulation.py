################################################################################
#
#    covid-simulation.py - Simulate the effects of social distancing in COVID19
#    usage: python3 covid-simulation.py 
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

import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import animation

# parameters of the simulation
N = 50        # number of inhabitants
dt = 0.5       # time step (a.u.) 
size = 80      # size of the town (occupancy = N/size**2)
frames = 1000  # duration of the simulation
recoT = 150    # recovery time (in a.u.)
P = 0.75       # probability to infect other people
save = True

# a class representing a person
class Person:
    _position = [0, 0]
    _velx = 0
    _vely = 0
    _town = ''
    _ill= 0
    _illtime = 0
    _recoveryTime = 0
    _P = 0
    _dieP = 0

    def __init__(self, P, initialP = 0.03, dieP = 0.045, recoveryTime = recoT):
        self._position[0] = np.random.uniform()
        self._position[1] = np.random.uniform()
        theta = np.random.uniform(0, 2*math.pi)
        self._velx = np.cos(theta)
        self._vely = np.sin(theta)
        self._P = initialP
        self.setAsIll()
        self._P = P
        self._dieP = dieP/recoT
        self._recoveryTime = recoT
    def isIll(self):
        ret = False
        if self._ill == 1:
            ret = True
        return ret
    def velocity(self):
        return [self._velx, self._vely]
    def position(self):
        return self._position
    def moveToTown(self, town):
        self._town = town
        s = town.size() - 1
        self._position = [self._position[0] * s, self._position[1] * s]
        town.putPerson(self)
    def invertVelocity(self):
        self._velx = -self._velx
        self._vely = -self._vely 
    def setAsIll(self):
        if (np.random.uniform() < self._P):
            self._ill = 1
            _illtime = 0
    def setAsHealthy(self):
        self._ill = 0
        self._illtime = 0
        # once recovered it cannot be infected anymore
        self._P = 0
    def tryToRecover(self):
        # check if it recovers
        if self.isIll():
            recoveryTime = self._recoveryTime + np.random.normal(scale = 15)
            if self._illtime > recoveryTime:
                self.setAsHealthy()
        else:
            self._illtime += 1
    def bounce(self):
        # check if it is at the border
        s = self._town.size() - 1
        if self._position[0] >= s:
            self._position[0] = s
            self._velx = -self._velx
        if self._position[0] < 0:
            self._position[0] = 0
            self._velx = -self._velx
        if self._position[1] >= s:
            self._position[1] = s
            self._vely = -self._vely
        if self._position[1] < 0:
            self._position[1] = 0
            self._vely = -self._vely
    def meet(self, p):
        if p != None:
            # the velocity of the CM
            (Vx, Vy) = (self._velx/2 + p._velx/2, self._vely/2 + p._vely/2)
            # the velocity of self in the CM after the collision
            (Vx1, Vy1) = (-(self._velx-Vx), -(self._vely-Vy))
            # the velocity of p in the CM after the collision
            (Vx2, Vy2) = (-(p._velx-Vx), -(p._vely-Vy))
            # the velocity of self in the laboratory
            (Vx1, Vy1) = (Vx1 + Vx, Vy1 + Vy)
            # the velocity of p in the laboratory
            (Vx2, Vy2) = (Vx2 + Vx, Vy2 + Vy)
            self._velx = Vx1
            self._vely = Vy1
            p._velx = Vx2
            p._vely = Vy2
            # check if self can transmit infection to p
            if self.isIll():
                p.setAsIll()
    def step(self, dt):
        # perform a simulation step
        self.tryToRecover()
        # remove it from the current position
        self._town.removePerson(self)
        # compute next position
        self._position[0] += dt*self._velx
        self._position[1] += dt*self._vely
        self.bounce()
        # move it to the current position
        self._town.putPerson(self)
        # check if there is anyone in the vicinity
        x = int(self._position[0])
        y = int(self._position[1])
        s = self._town.size() - 1
        for ix in [max(0, x - 1), x, min(s, x + 1)]:
            for iy in [max(0, y - 1), y, min(s, y +1)]:
                if ix != x and iy != y:
                    p = self._town.personAt(ix, iy)
                    self.meet(p)
        # see if it must die
        if self.isIll():
            r = np.random.uniform()
            if r < self._dieP:
                self._town.population().remove(self)

# a town class                        
class Town:
    _population = []
    _name = None
    _map = None

    def __init__(self, name, size):
        self._name = name
        self._map = [[None for x in range(size)] for y in range(size)]
    def add(self, person):
        self._population.append(person)
        person.moveToTown(self)
    def population(self):
        return self._population
    def name(self):
        return self._name
    def size(self):
        return len(self._map[0])
    def putPerson(self, person):
        x = int(person.position()[0])
        y = int(person.position()[1])
        self._map[x][y] = person
    def removePerson(self, person):
        x = int(person.position()[0])
        y = int(person.position()[1])
        self._map[x][y] = None
    def step(self, dt):
        for p in self._population:
            p.step(dt)
    def personAt(self, x, y):
        return self._map[x][y]
    def nIll(self):
        n = 0
        for pi in self._population:
            if pi.isIll():
                n += 1
        return n
    
t = Town('COVIDVille', size)
for i in range(N):
    t.add(Person(P))

print('======== ' + t.name())
p = t.population()
size = t.size()

t.step(dt)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0, 0.1, 1, 1])
ay = fig.add_axes([0.5, 0, 1, 0.1])
ax.set_xlim(0, size)
ax.set_ylim(0, size)
ax.get_xaxis().set_visible(False)
ay.set_xlim(0, frames / 0.5)
ay.set_ylim(0, N)
ay.get_xaxis().set_visible(False)
#ay.get_yaxis().set_visible(False)

x = []
y = []
p = t.population()
for pi in p:
    x.append(pi.position()[0])
for pi in p:
    y.append(pi.position()[1])
scat = ax.scatter(x, y, s=10, edgecolors='red', facecolors='red')
time = []
ills = []
time.append(0)
ills.append(t.nIll())
rills = ills[-1]/len(p)*100
txt = plt.text(0.01, 0.03,
               't: {} P: {} Infected: {} ({:.1f} %)'.format(1, len(p), ills, rills),
               fontsize = 10, transform=fig.transFigure)

def animate(i):
    t.step(dt)
    p = t.population()
    x = []
    y = []
    health = []
    for pi in p:
        x.append(pi.position()[0])
        y.append(pi.position()[1])
        if pi.isIll():
            health.append(0)
        else:
            health.append(20)
    scat.set_offsets(np.c_[x, y])
    scat.set_array(np.array(health))
    if i % (frames/100) == 0:
        time.append(i)
        ills.append(t.nIll())
    rills = 0
    if len(p) > 0:
        rills = ills[-1]/len(p)*100
    txt.set_text('t: {} P: {} Infected: {} ({:.1f} %)'.format(i + 1, len(p), ills[-1], rills))
    M = max(ills)
    ay.set_ylim(0, M*1.1)
    ay.plot(time, ills, 'b-')
    print('t: {:5d} p: {:6d} p/N: {:3.0f} % i/p: {:5.2f} %'.
          format(i, len(p), len(p)/N*100, ills[-1]/len(p)*100))
    return scat

anim = animation.FuncAnimation(fig, animate, interval=20, frames=frames, repeat = False)

if save:
    anim.save('covid_animation_n{}x{}.mp4'.format(N, size), fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
