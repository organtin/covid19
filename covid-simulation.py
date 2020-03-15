import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import animation

# a class representing a person
class Person:
    _position = [0, 0]
    _velx = 0
    _vely = 0
    _town = ''
    _ill = 0
    _illtime = 0
    _recoveryTime = 200
    _P = 0

    def __init__(self, P):
        self._position[0] = np.random.uniform()
        self._position[1] = np.random.uniform()
        theta = np.random.uniform(0, 2*math.pi)
        self._velx = np.cos(theta)
        self._vely = np.sin(theta)
        self._P = P
        self.setAsIll()
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
    def step(self, dt):
        # perform a simulation step
        # if it is ill increase the corresponding time
        if self.isIll():
            self._illtime += 1
        # check if it recovers
        recoveryTime = self._recoveryTime + np.random.normal(scale = 15)
        if self._illtime > recoveryTime:
            self.setAsHealthy()
        # remove it from the current position
        self._town.removePerson(self)
        # compute next position
        self._position[0] += dt*self._velx
        self._position[1] += dt*self._vely
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
        # move it to the current position
        self._town.putPerson(self)
        # check if there is anyone in the vicinity
        x = int(self._position[0])
        y = int(self._position[1])
        for ix in [max(0, x - 1), x, min(s, x + 1)]:
            for iy in [max(0, y - 1), y, min(s, y +1)]:
                if ix != x and iy != y:
                    p = self._town.personAt(ix, iy)
                    if p != None:
                        self.invertVelocity()
                        p.invertVelocity()
                        # check if it can transmit infection
                        if self.isIll():
                            p.setAsIll()

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
    
N = 1000
dt = 0.1
size = 200

t = Town('COVIDVille', size)
for i in range(N):
    t.add(Person(0.25))

print('======== ' + t.name())
p = t.population()
size = t.size()

t.step(dt)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0, 0.1, 1, 1])
ax.set_xlim(0, size)
ax.set_ylim(0, size)
ax.get_xaxis().set_visible(False)

x = []
y = []
p = t.population()
for pi in p:
    x.append(pi.position()[0])
for pi in p:
    y.append(pi.position()[1])
scat = ax.scatter(x, y, s=10, edgecolors='red', facecolors='red')
ills = t.nIll()
rills = ills/len(p)*100
txt = plt.text(0.1, 0.03, 'Infected: {} ({:.1f} %)'.format(ills, rills),
               fontsize = 13, transform=fig.transFigure)

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
    ills = t.nIll()
    rills = ills/len(p)*100
    txt.set_text('Infected: {} ({:.1f} %)'.format(ills, rills))
    return scat

anim = animation.FuncAnimation(fig, animate, interval=20, frames=200, repeat = False)

anim.save('covid_animation_n200x1000.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
