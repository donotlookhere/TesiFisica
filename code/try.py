import agama
import numpy as np
import matplotlib.pyplot as plt
import gala.potential
import gc

# units 1 M_sol, 1 kpc, km/s
agama.setUnits(mass=1, length=1, velocity=1)
G_constant=4.3 * pow(10,-6)
# try plummer potential
scaleRadius=2*pow(10,-3)
mass=pow(10,5)

# check AGAMA plummer model density against analytical formula
def plummerManualDensity(array):
    result=[]
    for column in array:
        radius= np.sqrt(pow(column[0], 2) + pow(column[1], 2) + pow(column[2], 2))
        result.append((3*mass/(4*np.pi*pow(scaleRadius,3)))*pow((1+pow(radius,2)/pow(scaleRadius,2)),-5/2))
    return result

# NOTE: this formula was written looking at gala's builtin_potentials.c file, not sure if it is actually correct
# https://github.com/adrn/gala/blob/main/gala/potential/potential/builtin/builtin_potentials.c#L245
def hernquistManualDensity(array):
    result=[]
    for column in array:
        radius= np.sqrt(pow(column[0], 2) + pow(column[1], 2) + pow(column[2], 2))
        rho0 = mass / (2 * np.pi * scaleRadius * scaleRadius * scaleRadius)
        result.append(rho0/((radius/scaleRadius)*pow(1+radius/scaleRadius,3)))
    return result

# AGAMA plummer density and potential
plummerDensity=agama.Density(type="Plummer",mass=mass,scaleRadius=scaleRadius)

plummer = agama.GalaPotential(type="Plummer",mass=mass,scaleRadius=scaleRadius)

# Hernquist (not present in AGAMA, we have to use the GALA one)
hernquistGala= gala.potential.HernquistPotential(m=mass,c=scaleRadius)
Hernquist=agama.GalaPotential(hernquistGala)

#Black hole potentials
#BH mass set at 10%
#BH scale radius set at 5%
bhMass=mass/9
scaleRadiusBH=scaleRadius/19

#kepler potential according to agama documentation
keplerPotential = agama.GalaPotential(type="Plummer",mass=bhMass,scaleRadius=0)
smallPlummer = agama.GalaPotential(type="Plummer",mass=bhMass,scaleRadius=scaleRadiusBH)

BHPotential1 = agama.Potential(plummer,keplerPotential)
BHPotential2 = agama.Potential(plummer,smallPlummer)


print(agama.getUnits())
# get points
t1 = np.arange(0.0, 10*pow(10,-3), pow(10,-4))
t2 = np.full(t1.size,0)
t3 = np.full(t1.size,0)
points=np.vstack((t1,t2,t3)).T


# plot density
plt.plot(t1,plummerManualDensity(points),label="Analytical Plummer")
plt.plot(t1,plummer.density(points.T),linestyle="--",label="Agama Plummer")
plt.plot(t1,hernquistManualDensity(points),label="Analytical Hernquist")
plt.plot(t1,Hernquist.density(points.T),linestyle="--",label="GALA Hernquist")
plt.ylabel(r'$\rho$')
plt.xlabel("$r$")
plt.legend()
plt.gca().set_yscale('log')
plt.title("Check density against known formulae")
plt.savefig("./output/densityCheck.png")
plt.show()
# clear plot
plt.clf()

vu = plummer.units['length'] / plummer.units['time']
points = np.ndarray(shape=(3,3),buffer=np.array([[0.0015,0.0015,0],[0.002,0.002,0],[0.01,0.02,0]])).T

print("Points:")
print(*points, sep='\n')

velocity = np.ndarray(shape=(3,3),buffer=np.array([[-7,10,0],[0.2,1,0],[5,2,0]]))
starting_points = np.hstack((points.T, velocity))
escapeVelocityPlummer=np.sqrt(-2*plummer.potential(points.T))
chosenVelocitySquared=[]
for vel in velocity:
    chosenVelocitySquared.append(pow(vel[0],2)+pow(vel[1],2)+pow(vel[2],2))
chosenVelocityModule=np.sqrt(chosenVelocitySquared)
print("Chosen velocity:")
print(*chosenVelocityModule, sep=' ')
print("Escape velocity (Plummer) on same points:")
print(*escapeVelocityPlummer, sep=' ')


result=agama.orbit(potential=plummer,ic=starting_points,time=[0.05,1,5],trajsize=[10000000,1000000,10000])

plt.plot(result[0][1][:,0], result[0][1][:,1],label="Orbit1")
plt.legend()
plt.title("Plummer orbits")
plt.savefig("./output/PlummerOrbits1.png")
plt.show()
# clear plot
plt.clf()

plt.plot(result[1][1][:,0], result[1][1][:,1],label="Orbit2")
plt.legend()
plt.title("Plummer orbits")
plt.savefig("./output/PlummerOrbits2.png")
plt.show()
# clear plot
plt.clf()

plt.plot(result[2][1][:,0], result[2][1][:,1],label="Orbit3")
plt.legend()
plt.title("Plummer orbits")
plt.savefig("./output/PlummerOrbits3.png")
plt.show()
# clear plot
plt.clf()

# find actions and energy
af = agama.ActionFinder(plummer, False)
t1=result[0][0]
t2=result[1][0]
t3=result[2][0]
actionsOrbit0 = af(result[0][1])
actionsOrbit1 = af(result[1][1])
actionsOrbit2 = af(result[2][1])
energyOrbit0 = np.add(plummer.potential(result[0][1][:,0:3]),0.5*np.sum(result[0][1][:, 3:6]**2, axis=1))
energyOrbit1 = np.add(plummer.potential(result[1][1][:,0:3]),0.5*np.sum(result[1][1][:, 3:6]**2, axis=1))
energyOrbit2 = np.add(plummer.potential(result[2][1][:,0:3]),0.5*np.sum(result[2][1][:, 3:6]**2, axis=1))

#plot energy and actions
plt.plot(t1, actionsOrbit0[:,2],label="Orbit1")
plt.plot(t2, actionsOrbit1[:,2],label="Orbit2")
plt.plot(t3, actionsOrbit2[:,2],label="Orbit3")
plt.legend()
plt.title("Plummer L_z")
plt.savefig("./output/PlummerL_z.png")
plt.show()
plt.clf()

plt.plot(t1, actionsOrbit0[:,1]+np.abs(actionsOrbit0[:,2]),label="Orbit1")
plt.plot(t2, actionsOrbit1[:,1]+np.abs(actionsOrbit1[:,2]),label="Orbit2")
plt.plot(t3, actionsOrbit2[:,1]+np.abs(actionsOrbit2[:,2]),label="Orbit3")
plt.legend()
plt.title("Total angular momentum L")
plt.savefig("./output/Plummer_angular_momentum.png")
plt.show()
plt.clf()

plt.plot(t1, energyOrbit0, label="Orbit1")
plt.plot(t3, energyOrbit2, label="Orbit3")
plt.plot(t2, energyOrbit1, label="Orbit2")
plt.legend()
plt.title("Energy")
plt.savefig("./output/PlummerEnergy.png")
plt.show()
plt.clf()

# clear interpreter memory
del(result)
gc.collect()

#BH integration 1
result=agama.orbit(potential=BHPotential1,ic=starting_points,time=[0.05,0.1,5],trajsize=[10000000,1000000,10000])

plt.plot(result[0][1][:,0], result[0][1][:,1],label="Orbit1")
plt.legend()
plt.title("Plummer+Kepler orbits")
plt.savefig("./output/BHKeplerOrbits1.png")
plt.show()
# clear plot
plt.clf()

plt.plot(result[1][1][:,0], result[1][1][:,1],label="Orbit2")
plt.legend()
plt.title("Plummer+Kepler orbits")
plt.savefig("./output/BHKeplerOrbits2.png")
plt.show()
# clear plot
plt.clf()

plt.plot(result[2][1][:,0], result[2][1][:,1],label="Orbit3")
plt.legend()
plt.title("Plummer+Kepler orbits")
plt.savefig("./output/BHKeplerOrbits3.png")
plt.show()
# clear plot
plt.clf()

# find actions and energy
af = agama.ActionFinder(BHPotential1, False)
t1=result[0][0]
t2=result[1][0]
t3=result[2][0]
actionsOrbit0 = af(result[0][1])
actionsOrbit1 = af(result[1][1])
actionsOrbit2 = af(result[2][1])
energyOrbit0 = np.add(BHPotential1.potential(result[0][1][:,0:3]),0.5*np.sum(result[0][1][:, 3:6]**2, axis=1))
energyOrbit1 = np.add(BHPotential1.potential(result[1][1][:,0:3]),0.5*np.sum(result[1][1][:, 3:6]**2, axis=1))
energyOrbit2 = np.add(BHPotential1.potential(result[2][1][:,0:3]),0.5*np.sum(result[2][1][:, 3:6]**2, axis=1))

#plot energy and actions
plt.plot(t1, actionsOrbit0[:,2],label="Orbit1")
plt.plot(t2, actionsOrbit1[:,2],label="Orbit2")
plt.plot(t3, actionsOrbit2[:,2],label="Orbit3")
plt.legend()
plt.title("Plummer+Kepler L_z")
plt.savefig("./output/BHKeplerL_z.png")
plt.show()
plt.clf()

plt.plot(t1, actionsOrbit0[:,1]+np.abs(actionsOrbit0[:,2]),label="Orbit1")
plt.plot(t2, actionsOrbit1[:,1]+np.abs(actionsOrbit1[:,2]),label="Orbit2")
plt.plot(t3, actionsOrbit2[:,1]+np.abs(actionsOrbit2[:,2]),label="Orbit3")
plt.legend()
plt.title("Total angular momentum L")
plt.savefig("./output/BHKepler_angular_momentum.png")
plt.show()
plt.clf()

plt.plot(t1, energyOrbit0, label="Orbit1")
plt.plot(t3, energyOrbit2, label="Orbit3")
plt.plot(t2, energyOrbit1, label="Orbit2")

plt.legend()
plt.title("Energy")
plt.savefig("./output/BHKeplerEnergy.png")
plt.show()
plt.clf()

# clear interpreter memory
del(result)
gc.collect()

#BH integration 2
result=agama.orbit(potential=BHPotential2,ic=starting_points,time=[0.05,0.1,5],trajsize=[10000000,1000000,10000])

plt.plot(result[0][1][:,0], result[0][1][:,1],label="Orbit1")
plt.legend()
plt.title("Plummer+SmallPlummer orbits")
plt.savefig("./output/BHSmallPlummerOrbits1.png")
plt.show()
# clear plot
plt.clf()

plt.plot(result[1][1][:,0], result[1][1][:,1],label="Orbit2")
plt.legend()
plt.title("Plummer+SmallPlummer orbits")
plt.savefig("./output/BHSmallPlummerOrbits2.png")
plt.show()
# clear plot
plt.clf()

plt.plot(result[2][1][:,0], result[2][1][:,1],label="Orbit3")
plt.legend()
plt.title("Plummer+SmallPlummer orbits")
plt.savefig("./output/BHSmallPlummerOrbits3.png")
plt.show()
# clear plot
plt.clf()

# find actions and energy
af = agama.ActionFinder(BHPotential2, False)
t1=result[0][0]
t2=result[1][0]
t3=result[2][0]
actionsOrbit0 = af(result[0][1])
actionsOrbit1 = af(result[1][1])
actionsOrbit2 = af(result[2][1])
energyOrbit0 = np.add(BHPotential2.potential(result[0][1][:,0:3]),0.5*np.sum(result[0][1][:, 3:6]**2, axis=1))
energyOrbit1 = np.add(BHPotential2.potential(result[1][1][:,0:3]),0.5*np.sum(result[1][1][:, 3:6]**2, axis=1))
energyOrbit2 = np.add(BHPotential2.potential(result[2][1][:,0:3]),0.5*np.sum(result[2][1][:, 3:6]**2, axis=1))

#plot energy and actions
plt.plot(t1, actionsOrbit0[:,2],label="Orbit1")
plt.plot(t2, actionsOrbit1[:,2],label="Orbit2")
plt.plot(t3, actionsOrbit2[:,2],label="Orbit3")
plt.legend()
plt.title("Plummer+SmallPlummer L_z")
plt.savefig("./output/BHSmallPlummerL_z.png")
plt.show()
plt.clf()

plt.plot(t1, actionsOrbit0[:,1]+np.abs(actionsOrbit0[:,2]),label="Orbit1")
plt.plot(t2, actionsOrbit1[:,1]+np.abs(actionsOrbit1[:,2]),label="Orbit2")
plt.plot(t3, actionsOrbit2[:,1]+np.abs(actionsOrbit2[:,2]),label="Orbit3")
plt.legend()
plt.title("Total angular momentum L")
plt.savefig("./output/BHSmallPlummer_angular_momentum.png")
plt.show()
plt.clf()

plt.plot(t1, energyOrbit0, label="Orbit1")
plt.plot(t3, energyOrbit2, label="Orbit3")
plt.plot(t2, energyOrbit1, label="Orbit2")

plt.legend()
plt.title("Energy")
plt.savefig("./output/BHSmallPlummerEnergy.png")
plt.show()
plt.clf()

# clear interpreter memory
del(result)
gc.collect()



#Hernquist integration

result=Hernquist.integrate_orbit(starting_points[0].T, dt=0.00000001, n_steps=10000, Integrator=gala.integrate.DOPRI853Integrator)
result1=Hernquist.integrate_orbit(starting_points[1].T, dt=0.000001, n_steps=10000, Integrator=gala.integrate.DOPRI853Integrator)
result2=Hernquist.integrate_orbit(starting_points[2].T, dt=0.0000005, n_steps=10000, Integrator=gala.integrate.DOPRI853Integrator)

plt.plot(result.x, result.y,label="Orbit1")
plt.legend()
plt.title("Hernquist orbits")
plt.savefig("./output/HernquistOrbits1.png")
plt.show()
# clear plot
plt.clf()

plt.plot(result1.x, result1.y,label="Orbit2")
plt.legend()
plt.title("Hernquist orbits")
plt.savefig("./output/HernquistOrbits2.png")
plt.show()
# clear plot
plt.clf()

plt.plot(result2.x, result2.y,label="Orbit3")
plt.legend()
plt.title("Hernquist orbits")
plt.savefig("./output/HernquistOrbits3.png")
plt.show()
# clear plot
plt.clf()

# find actions and energy
t1=result.t
t2=result1.t
t3=result2.t
momentumModulo0=[]
momentumModulo1=[]
momentumModulo2=[]
momentumZ0=[]
momentumZ1=[]
momentumZ2=[]
for results,momentums,momentumsz in zip([result,result1,result2],[momentumModulo0,momentumModulo1,momentumModulo2],[momentumZ0,momentumZ1,momentumZ2]):
    for x, y, z, v_x, v_y, v_z in zip(results.x, results.y, results.z,results.v_x,results.v_y,results.v_z):
        m_vec=np.cross(np.array([x,y,z]),np.array([v_x,v_y,v_z]))
        momentumsz.append(np.linalg.norm(np.dot(m_vec,np.array([0,0,1]))))
        momentums.append(np.linalg.norm(m_vec))

energyOrbit0 = np.add(Hernquist.potential(np.array([result.x,result.y,result.z]).T),0.5*np.sum(np.array([result.v_x,result.v_y,result.v_z]).T**2, axis=1))
energyOrbit1 = np.add(Hernquist.potential(np.array([result1.x,result1.y,result1.z]).T),0.5*np.sum(np.array([result1.v_x,result1.v_y,result1.v_z]).T**2, axis=1))
energyOrbit2 = np.add(Hernquist.potential(np.array([result2.x,result2.y,result2.z]).T),0.5*np.sum(np.array([result2.v_x,result2.v_y,result2.v_z]).T**2, axis=1))

#plot energy and actions
plt.plot(t1, momentumZ0,label="Orbit1")
plt.plot(t2, momentumZ1,label="Orbit2")
plt.plot(t3, momentumZ2,label="Orbit3")
plt.legend()
plt.title("Hernquist L_z")
plt.savefig("./output/HernquistL_z.png")
plt.show()
plt.clf()

plt.plot(t1, momentumModulo0,label="Orbit1")
plt.plot(t2, momentumModulo1,label="Orbit2")
plt.plot(t3, momentumModulo2,label="Orbit3")
plt.legend()
plt.title("Total angular momentum L")
plt.savefig("./output/Hernquist_angular_momentum.png")
plt.show()
plt.clf()

plt.plot(t1, energyOrbit0, label="Orbit1")
plt.plot(t3, energyOrbit2, label="Orbit3")
plt.plot(t2, energyOrbit1, label="Orbit2")

plt.legend()
plt.title("Energy")
plt.savefig("./output/HernquistEnergy.png")
plt.show()
plt.clf()

# clear interpreter memory
del(result)
gc.collect()