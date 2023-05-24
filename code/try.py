import agama
import numpy as np
import matplotlib.pyplot as plt
import gala.potential
import gc
import matplotlib
from gala.units import UnitSystem
import astropy.units as u
matplotlib.use('Agg')

# units 1 M_sol, 1 kpc, km/s
agama.setUnits(mass=1, length=1, velocity=1)
usys = UnitSystem(u.kpc, 977.79222168*u.Myr, u.Msun, u.radian, u.km/u.s)
G_constant=4.3 * pow(10,-6)
# try plummer potential
scaleRadius=2*pow(10,-3)
mass=pow(10,5)
def relativeVariation(array):
    return np.divide(np.abs(np.subtract(array,np.full(np.shape(array),array[0]))),np.abs(np.full(np.shape(array),array[0])))
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
plummerGALA=gala.potential.PlummerPotential(m=mass, b=scaleRadius,units=usys)

# Hernquist (not present in AGAMA, we have to use the GALA one)
hernquistGala= gala.potential.HernquistPotential(m=mass,c=scaleRadius,units=usys)
Hernquist=agama.GalaPotential(hernquistGala)

#Black hole potentials
#BH mass set at 10%
#BH scale radius set at 5%
bhMass=mass/9
scaleRadiusBH=scaleRadius/19

#kepler potential according to agama documentation
keplerPotential = agama.GalaPotential(type="Plummer",mass=bhMass,scaleRadius=0)
smallPlummer = agama.GalaPotential(type="Plummer",mass=bhMass,scaleRadius=scaleRadiusBH)
keplerGALA = gala.potential.KeplerPotential(m=bhMass,units=usys)
smallPlummerGALA = gala.potential.PlummerPotential(m=bhMass,b=scaleRadiusBH,units=usys)
BHPotential1GALA = plummerGALA + keplerGALA
BHPotential2GALA = plummerGALA + smallPlummerGALA
print(BHPotential2GALA.__class__.__name__)
print(BHPotential1GALA.__class__.__name__)


BHPotential1 = agama.GalaPotential(plummer,keplerPotential)
BHPotential2 = agama.GalaPotential(plummer,smallPlummer)


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
plt.savefig("./output/try/densityCheck.png")
#plt.show()
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

potentialsAGAMA=[plummer,BHPotential1,BHPotential2,plummer,BHPotential1,BHPotential2]
timesAGAMA=[[0.05,1,5],[0.05,0.1,5],[0.05,0.1,5],[2,2,2],[2,2,2],[2,2,2]]
trajsizesAGAMA=[[10000000,1000000,10000],
                [10000000,1000000,10000],
                [10000000,1000000,10000],
                [10000000,10000000,10000000],
                [10000000,10000000,10000000],
                [10000000,10000000,10000000]]
titlesAGAMA=[["Plummer orbits","Plummer orbits","Plummer orbits","Plummer L_z","Total angular momentum L","Energy","Plummer L_z relative variation","Total angular momentum L relative variation","Energy relative variation"],
             ["Plummer+Kepler orbits","Plummer+Kepler orbits","Plummer+Kepler orbits","Plummer+Kepler L_z","Total angular momentum L","Energy","Plummer+Kepler L_z relative variation","Total angular momentum L relative variation","Energy relative variation"],
             ["Plummer+SmallPlummer orbits","Plummer+SmallPlummer orbits","Plummer+SmallPlummer orbits","Plummer+SmallPlummer L_z","Total angular momentum L","Energy","Plummer+SmallPlummer L_z relative variation","Total angular momentum L relative variation","Energy relative variation"],
             ["Plummer orbits", "Plummer orbits", "Plummer orbits", "Plummer L_z", "Total angular momentum L", "Energy","Plummer L_z relative variation", "Total angular momentum L relative variation","Energy relative variation"],
             ["Plummer+Kepler orbits", "Plummer+Kepler orbits", "Plummer+Kepler orbits", "Plummer+Kepler L_z","Total angular momentum L", "Energy", "Plummer+Kepler L_z relative variation","Total angular momentum L relative variation", "Energy relative variation"],
             ["Plummer+SmallPlummer orbits", "Plummer+SmallPlummer orbits", "Plummer+SmallPlummer orbits","Plummer+SmallPlummer L_z", "Total angular momentum L", "Energy","Plummer+SmallPlummer L_z relative variation", "Total angular momentum L relative variation","Energy relative variation"]]
filenamesAGAMA=[["./output/try/differentTime/PlummerOrbits1.png","./output/try/differentTime/PlummerOrbits2.png","./output/try/differentTime/PlummerOrbits3.png","./output/try/differentTime/PlummerL_z.png","./output/try/differentTime/Plummer_angular_momentum.png","./output/try/differentTime/PlummerEnergy.png","./output/try/differentTime/PlummerL_zRel.png","./output/try/differentTime/Plummer_angular_momentumRel.png","./output/try/differentTime/PlummerEnergyRel.png"],
                ["./output/try/differentTime/BHKeplerOrbits1.png","./output/try/differentTime/BHKeplerOrbits2.png","./output/try/differentTime/BHKeplerOrbits3.png","./output/try/differentTime/BHKeplerL_z.png","./output/try/differentTime/BHKepler_angular_momentum.png","./output/try/differentTime/BHKeplerEnergy.png","./output/try/differentTime/BHKeplerL_zRel.png","./output/try/differentTime/BHKepler_angular_momentumRel.png","./output/try/differentTime/BHKeplerEnergyRel.png"],
                ["./output/try/differentTime/BHSmallPlummerOrbits1.png","./output/try/differentTime/BHSmallPlummerOrbits2.png","./output/try/differentTime/BHSmallPlummerOrbits3.png","./output/try/differentTime/BHSmallPlummerL_z.png","./output/try/differentTime/BHSmallPlummer_angular_momentum.png","./output/try/differentTime/BHSmallPlummerEnergy.png","./output/try/differentTime/BHSmallPlummerL_zRel.png","./output/try/differentTime/BHSmallPlummer_angular_momentumRel.png","./output/try/differentTime/BHSmallPlummerEnergyRel.png"],
                ["./output/try/equalTime/PlummerOrbits1.png", "./output/try/equalTime/PlummerOrbits2.png","./output/try/equalTime/PlummerOrbits3.png","./output/try/equalTime/PlummerL_z.png","./output/try/equalTime/Plummer_angular_momentum.png","./output/try/equalTime/PlummerEnergy.png", "./output/try/equalTime/PlummerL_zRel.png","./output/try/equalTime/Plummer_angular_momentumRel.png","./output/try/equalTime/PlummerEnergyRel.png"],
                ["./output/try/equalTime/BHKeplerOrbits1.png", "./output/try/equalTime/BHKeplerOrbits2.png","./output/try/equalTime/BHKeplerOrbits3.png", "./output/try/equalTime/BHKeplerL_z.png","./output/try/equalTime/BHKepler_angular_momentum.png","./output/try/equalTime/BHKeplerEnergy.png", "./output/try/equalTime/BHKeplerL_zRel.png","./output/try/equalTime/BHKepler_angular_momentumRel.png","./output/try/equalTime/BHKeplerEnergyRel.png"],
                ["./output/try/equalTime/BHSmallPlummerOrbits1.png","./output/try/equalTime/BHSmallPlummerOrbits2.png","./output/try/equalTime/BHSmallPlummerOrbits3.png","./output/try/equalTime/BHSmallPlummerL_z.png","./output/try/equalTime/BHSmallPlummer_angular_momentum.png","./output/try/equalTime/BHSmallPlummerEnergy.png","./output/try/equalTime/BHSmallPlummerL_zRel.png","./output/try/equalTime/BHSmallPlummer_angular_momentumRel.png","./output/try/equalTime/BHSmallPlummerEnergyRel.png"]]

for selectedPotential,selectedTime,selectedTrajSize,selectedTitle,selectedFilename in zip(potentialsAGAMA[3:],timesAGAMA[3:],trajsizesAGAMA[3:],titlesAGAMA[3:],filenamesAGAMA[3:]):
    result = agama.orbit(potential=selectedPotential, ic=starting_points, time=selectedTime, trajsize=selectedTrajSize,accuracy=1e-16,dtype="float64")

    plt.plot(result[0][1][:, 0], result[0][1][:, 1], label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[0])
    plt.axis('equal')
    plt.savefig(selectedFilename[0])
    plt.show()
    # clear plot
    plt.clf()

    plt.plot(result[1][1][:, 0], result[1][1][:, 1], label="Orbit2")
    plt.legend()
    plt.title(selectedTitle[1])
    plt.axis('equal')
    plt.savefig(selectedFilename[1])
    plt.show()
    # clear plot
    plt.clf()

    plt.plot(result[2][1][:, 0], result[2][1][:, 1], label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[2])
    plt.axis('equal')
    plt.savefig(selectedFilename[2])
    plt.show()
    # clear plot
    plt.clf()
    gc.collect()

    # find actions and energy
    t1 = result[0][0]
    t2 = result[1][0]
    t3 = result[2][0]
    momentumModulo0 = []
    momentumModulo1 = []
    momentumModulo2 = []
    momentumZ0 = []
    momentumZ1 = []
    momentumZ2 = []
    for results, momentums, momentumsz in zip([result[0], result[1], result[2]],
                                             [momentumModulo0, momentumModulo1, momentumModulo2],
                                                  [momentumZ0, momentumZ1, momentumZ2]):
        m_vec = np.cross(results[1][:, 0:3], results[1][:, 3:6])
        momentumsz.append(np.einsum('ij,ij->i',m_vec, np.full((np.shape(m_vec)),np.array([0, 0, 1]))))
        momentums.append(np.linalg.norm(m_vec,axis=1))

    energyOrbit0 = np.add(selectedPotential.potential(result[0][1][:, 0:3]),
                          0.5 * np.sum(result[0][1][:, 3:6] ** 2, axis=1))
    energyOrbit1 = np.add(selectedPotential.potential(result[1][1][:, 0:3]),
                          0.5 * np.sum(result[1][1][:, 3:6] ** 2, axis=1))
    energyOrbit2 = np.add(selectedPotential.potential(result[2][1][:, 0:3]),
                          0.5 * np.sum(result[2][1][:, 3:6] ** 2, axis=1))
    del(result)
    gc.collect()
    # plot energy and actions
    plt.plot(t1, momentumZ0[0], label="Orbit1")
    plt.plot(t2, momentumZ1[0], label="Orbit2")
    plt.plot(t3, momentumZ2[0], label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[3])
    plt.savefig(selectedFilename[3])
    plt.show()
    plt.clf()
    gc.collect()

    # plot energy and actions
    plt.plot(t3, relativeVariation(momentumZ2[0]), label="Orbit3")
    plt.plot(t2, relativeVariation(momentumZ1[0]), label="Orbit2")
    plt.plot(t1, relativeVariation(momentumZ0[0]), label="Orbit1")

    plt.legend()
    plt.title(selectedTitle[6])
    plt.savefig(selectedFilename[6])
    plt.show()
    plt.clf()
    gc.collect()


    plt.plot(t1, momentumModulo0[0], label="Orbit1")
    plt.plot(t2, momentumModulo1[0], label="Orbit2")
    plt.plot(t3, momentumModulo2[0], label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[4])
    plt.savefig(selectedFilename[4])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t3, relativeVariation(momentumModulo2[0]), label="Orbit3")
    plt.plot(t2, relativeVariation(momentumModulo1[0]), label="Orbit2")
    plt.plot(t1, relativeVariation(momentumModulo0[0]), label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[7])
    plt.savefig(selectedFilename[7])
    plt.show()
    plt.clf()
    gc.collect()


    plt.plot(t1, energyOrbit0, label="Orbit1")
    plt.plot(t3, energyOrbit2, label="Orbit3")
    plt.plot(t2, energyOrbit1, label="Orbit2")
    plt.legend()
    plt.title(selectedTitle[5])
    plt.savefig(selectedFilename[5])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t3, relativeVariation(energyOrbit2), label="Orbit3")
    plt.plot(t2, relativeVariation(energyOrbit1), label="Orbit2")
    plt.plot(t1, relativeVariation(energyOrbit0), label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[8])
    plt.savefig(selectedFilename[8])
    plt.show()
    plt.clf()
    # clear interpreter memory
    gc.collect()

filenamesAGAMA=[["./output/try/differentTime/PlummerOrbits1.png","./output/try/differentTime/PlummerOrbits2.png","./output/try/differentTime/PlummerOrbits3.png","./output/try/differentTime/PlummerL_z.png","./output/try/differentTime/Plummer_angular_momentum.png","./output/try/differentTime/PlummerEnergy.png","./output/try/differentTime/PlummerL_zRel.png","./output/try/differentTime/Plummer_angular_momentumRel.png","./output/try/differentTime/PlummerEnergyRel.png"],
                ["./output/try/differentTime/BHKeplerOrbits1.png","./output/try/differentTime/BHKeplerOrbits2.png","./output/try/differentTime/BHKeplerOrbits3.png","./output/try/differentTime/BHKeplerL_z.png","./output/try/differentTime/BHKepler_angular_momentum.png","./output/try/differentTime/BHKeplerEnergy.png","./output/try/differentTime/BHKeplerL_zRel.png","./output/try/differentTime/BHKepler_angular_momentumRel.png","./output/try/differentTime/BHKeplerEnergyRel.png"],
                ["./output/try/differentTime/BHSmallPlummerOrbits1.png","./output/try/differentTime/BHSmallPlummerOrbits2.png","./output/try/differentTime/BHSmallPlummerOrbits3.png","./output/try/differentTime/BHSmallPlummerL_z.png","./output/try/differentTime/BHSmallPlummer_angular_momentum.png","./output/try/differentTime/BHSmallPlummerEnergy.png","./output/try/differentTime/BHSmallPlummerL_zRel.png","./output/try/differentTime/BHSmallPlummer_angular_momentumRel.png","./output/try/differentTime/BHSmallPlummerEnergyRel.png"],
                ["./output/try/symplectic/PlummerOrbits1.png", "./output/try/symplectic/PlummerOrbits2.png","./output/try/symplectic/PlummerOrbits3.png","./output/try/symplectic/PlummerL_z.png","./output/try/symplectic/Plummer_angular_momentum.png","./output/try/symplectic/PlummerEnergy.png", "./output/try/symplectic/PlummerL_zRel.png","./output/try/symplectic/Plummer_angular_momentumRel.png","./output/try/symplectic/PlummerEnergyRel.png"],
                ["./output/try/symplectic/BHKeplerOrbits1.png", "./output/try/symplectic/BHKeplerOrbits2.png","./output/try/symplectic/BHKeplerOrbits3.png", "./output/try/symplectic/BHKeplerL_z.png","./output/try/symplectic/BHKepler_angular_momentum.png","./output/try/symplectic/BHKeplerEnergy.png", "./output/try/symplectic/BHKeplerL_zRel.png","./output/try/symplectic/BHKepler_angular_momentumRel.png","./output/try/symplectic/BHKeplerEnergyRel.png"],
                ["./output/try/symplectic/BHSmallPlummerOrbits1.png","./output/try/symplectic/BHSmallPlummerOrbits2.png","./output/try/symplectic/BHSmallPlummerOrbits3.png","./output/try/symplectic/BHSmallPlummerL_z.png","./output/try/symplectic/BHSmallPlummer_angular_momentum.png","./output/try/symplectic/BHSmallPlummerEnergy.png","./output/try/symplectic/BHSmallPlummerL_zRel.png","./output/try/symplectic/BHSmallPlummer_angular_momentumRel.png","./output/try/symplectic/BHSmallPlummerEnergyRel.png"]]
integratorAGAMA=[[gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator],
                [gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator],
                 [gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator],
                 [gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator],
                 [gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator],
                 [gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator]]
potentialsAGAMA=[[plummerGALA,plummerGALA,plummerGALA],
                 [BHPotential1GALA,BHPotential1GALA,BHPotential1GALA],
                 [BHPotential2GALA,BHPotential2GALA,BHPotential2GALA],
                 [plummerGALA,plummerGALA,plummerGALA],
                 [BHPotential1GALA,BHPotential1GALA,BHPotential1GALA],
                 [BHPotential2GALA,BHPotential2GALA,BHPotential2GALA]]

for selectedPotential,selectedTime,selectedTrajSize,selectedTitle,selectedFilename,selectedIntegrator in zip(potentialsAGAMA[3:],timesAGAMA[3:],trajsizesAGAMA[3:],titlesAGAMA[3:],filenamesAGAMA[3:],integratorAGAMA[3:]):    #Symplectic integration
    result = selectedPotential[0].integrate_orbit(starting_points[0].T, dt=selectedTime[0]/selectedTrajSize[0], n_steps=selectedTrajSize[0],
                                       Integrator=selectedIntegrator[0],cython_if_possible=True)
    result1 = selectedPotential[1].integrate_orbit(starting_points[1].T, dt=selectedTime[1]/selectedTrajSize[1], n_steps=selectedTrajSize[1],
                                        Integrator=selectedIntegrator[1],cython_if_possible=True)
    result2 = selectedPotential[2].integrate_orbit(starting_points[2].T, dt=selectedTime[2]/selectedTrajSize[2], n_steps=selectedTrajSize[2],
                                        Integrator=selectedIntegrator[2],cython_if_possible=True)

    plt.plot(result.x, result.y, label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[0])
    plt.axis('equal')
    plt.show()
    plt.savefig(selectedFilename[0])
    # clear plot
    plt.clf()
    gc.collect()

    plt.plot(result1.x, result1.y, label="Orbit2")
    plt.legend()
    plt.title(selectedTitle[1])
    plt.axis('equal')
    plt.savefig(selectedFilename[1])
    #plt.show()
    # clear plot
    plt.clf()
    gc.collect()

    plt.plot(result2.x, result2.y, label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[2])
    plt.axis('equal')
    plt.savefig(selectedFilename[2])
    #plt.show()
    # clear plot
    plt.clf()
    gc.collect()

    # find actions and energy
    t1 = result.t
    t2 = result1.t
    t3 = result2.t
    momentumModulo0 = []
    momentumModulo1 = []
    momentumModulo2 = []
    momentumZ0 = []
    momentumZ1 = []
    momentumZ2 = []
    for results, momentums, momentumsz in zip([result, result1, result2],
                                              [momentumModulo0, momentumModulo1, momentumModulo2],
                                              [momentumZ0, momentumZ1, momentumZ2]):
        m_vec = np.cross(np.stack((results.x,results.y,results.z), axis = 1), np.stack((results.v_x,results.v_y,results.v_z), axis = 1))
        momentumsz.append(np.einsum('ij,ij->i', m_vec, np.full((np.shape(m_vec)), np.array([0, 0, 1]))))
        momentums.append(np.linalg.norm(m_vec,axis=1))

    energyOrbit0 = np.add(selectedPotential[0].energy(np.array([result.x, result.y, result.z])).to_value(),
                          0.5 * np.sum(np.array([result.v_x, result.v_y, result.v_z]).T ** 2, axis=1))
    energyOrbit1 = np.add(selectedPotential[1].energy(np.array([result1.x, result1.y, result1.z])).to_value(),
                          0.5 * np.sum(np.array([result1.v_x, result1.v_y, result1.v_z]).T ** 2, axis=1))
    energyOrbit2 = np.add(selectedPotential[2].energy(np.array([result2.x, result2.y, result2.z])).to_value(),
                          0.5 * np.sum(np.array([result2.v_x, result2.v_y, result2.v_z]).T ** 2, axis=1))

    del (result)
    gc.collect()

    # plot energy and actions
    plt.plot(t1, momentumZ0[0], label="Orbit1")
    plt.plot(t2, momentumZ1[0], label="Orbit2")
    plt.plot(t3, momentumZ2[0], label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[3])
    plt.savefig(selectedFilename[3])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t1, momentumModulo0[0], label="Orbit1")
    plt.plot(t2, momentumModulo1[0], label="Orbit2")
    plt.plot(t3, momentumModulo2[0], label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[4])
    plt.savefig(selectedFilename[4])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t1, energyOrbit0, label="Orbit1")
    plt.plot(t3, energyOrbit2, label="Orbit3")
    plt.plot(t2, energyOrbit1, label="Orbit2")

    plt.legend()
    plt.title(selectedTitle[5])
    plt.savefig(selectedFilename[5])
    plt.show()
    plt.clf()
    gc.collect()

    # plot energy and actions
    plt.plot(t3, relativeVariation(momentumZ2[0].to_value()), label="Orbit3")
    plt.plot(t2, relativeVariation(momentumZ1[0].to_value()), label="Orbit2")
    plt.plot(t1, relativeVariation(momentumZ0[0].to_value()), label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[6])
    plt.savefig(selectedFilename[6])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t3, relativeVariation(momentumModulo2[0].to_value()), label="Orbit3")
    plt.plot(t2, relativeVariation(momentumModulo1[0].to_value()), label="Orbit2")
    plt.plot(t1, relativeVariation(momentumModulo0[0].to_value()), label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[7])
    plt.savefig(selectedFilename[7])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t3, relativeVariation(energyOrbit2), label="Orbit3")
    plt.plot(t2, relativeVariation(energyOrbit1), label="Orbit2")
    plt.plot(t1, relativeVariation(energyOrbit0), label="Orbit1")

    plt.legend()
    plt.title(selectedTitle[8])
    plt.savefig(selectedFilename[8])
    plt.show()
    plt.clf()
    # clear interpreter memory
    gc.collect()

# ########
potentialsGALA=[[hernquistGala,hernquistGala,hernquistGala],
                [hernquistGala,hernquistGala,hernquistGala],
                [hernquistGala,hernquistGala,hernquistGala]]
timesGALA=[[2,2,2],[2,2,2],[2,2,2]]
trajsizesGALA=[[10000000,10000000,10000000],
                [10000000,10000000,10000000],
                [10000000,10000000,10000000]]
integratorGALA=[[gala.integrate.DOPRI853Integrator,gala.integrate.DOPRI853Integrator,gala.integrate.DOPRI853Integrator],
                [gala.integrate.DOPRI853Integrator,gala.integrate.DOPRI853Integrator,gala.integrate.DOPRI853Integrator],
                [gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator]]
titlesGALA=[["Hernquist orbits","Hernquist orbits","Hernquist orbits","Hernquist L_z","Total angular momentum L","Energy","Hernquist L_z relative variation","Total angular momentum L relative variation","Energy relative variation"],
            ["Hernquist orbits","Hernquist orbits","Hernquist orbits","Hernquist L_z","Total angular momentum L","Energy","Hernquist L_z relative variation","Total angular momentum L relative variation","Energy relative variation"],
            ["Hernquist orbits","Hernquist orbits","Hernquist orbits","Hernquist L_z","Total angular momentum L","Energy","Hernquist L_z relative variation","Total angular momentum L relative variation","Energy relative variation"]]
filenamesGALA=[["./output/try/differentTime/HernquistOrbits1.png","./output/try/differentTime/HernquistOrbits2.png","./output/try/differentTime/HernquistOrbits3.png","./output/try/differentTime/HernquistL_z.png","./output/try/differentTime/Hernquist_angular_momentum.png","./output/try/differentTime/HernquistEnergy.png","./output/try/differentTime/HernquistL_zRel.png","./output/try/differentTime/Hernquist_angular_momentumRel.png","./output/try/differentTime/HernquistEnergyRel.png"],
               ["./output/try/equalTime/HernquistOrbits1.png","./output/try/equalTime/HernquistOrbits2.png","./output/try/equalTime/HernquistOrbits3.png","./output/try/equalTime/HernquistL_z.png","./output/try/equalTime/Hernquist_angular_momentum.png","./output/try/equalTime/HernquistEnergy.png","./output/try/equalTime/HernquistL_zRel.png","./output/try/equalTime/Hernquist_angular_momentumRel.png","./output/try/equalTime/HernquistEnergyRel.png"],
               ["./output/try/symplectic/HernquistOrbits1.png","./output/try/symplectic/HernquistOrbits2.png","./output/try/symplectic/HernquistOrbits3.png","./output/try/symplectic/HernquistL_z.png","./output/try/symplectic/Hernquist_angular_momentum.png","./output/try/symplectic/HernquistEnergy.png","./output/try/symplectic/HernquistL_zRel.png","./output/try/symplectic/Hernquist_angular_momentumRel.png","./output/try/symplectic/HernquistEnergyRel.png"]]


for selectedPotential,selectedTime,selectedTrajSize,selectedTitle,selectedFilename,selectedIntegrator in zip(potentialsGALA[1:],timesGALA[1:],trajsizesGALA[1:],titlesGALA[1:],filenamesGALA[1:],integratorGALA[1:]):    #Symplectic integration
    result = selectedPotential[0].integrate_orbit(starting_points[0].T, dt=selectedTime[0]/selectedTrajSize[0], n_steps=selectedTrajSize[0],
                                       Integrator=selectedIntegrator[0],cython_if_possible=True)
    result1 = selectedPotential[1].integrate_orbit(starting_points[1].T, dt=selectedTime[1]/selectedTrajSize[1], n_steps=selectedTrajSize[1],
                                        Integrator=selectedIntegrator[1],cython_if_possible=True)
    result2 = selectedPotential[2].integrate_orbit(starting_points[2].T, dt=selectedTime[2]/selectedTrajSize[2], n_steps=selectedTrajSize[2],
                                        Integrator=selectedIntegrator[2],cython_if_possible=True)

    plt.plot(result.x, result.y, label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[0])
    plt.axis('equal')
    plt.show()
    plt.savefig(selectedFilename[0])
    # clear plot
    plt.clf()
    gc.collect()

    plt.plot(result1.x, result1.y, label="Orbit2")
    plt.legend()
    plt.title(selectedTitle[1])
    plt.axis('equal')
    plt.savefig(selectedFilename[1])
    #plt.show()
    # clear plot
    plt.clf()
    gc.collect()

    plt.plot(result2.x, result2.y, label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[2])
    plt.axis('equal')
    plt.savefig(selectedFilename[2])
    #plt.show()
    # clear plot
    plt.clf()
    gc.collect()

    # find actions and energy
    t1 = result.t
    t2 = result1.t
    t3 = result2.t
    momentumModulo0 = []
    momentumModulo1 = []
    momentumModulo2 = []
    momentumZ0 = []
    momentumZ1 = []
    momentumZ2 = []
    for results, momentums, momentumsz in zip([result, result1, result2],
                                              [momentumModulo0, momentumModulo1, momentumModulo2],
                                              [momentumZ0, momentumZ1, momentumZ2]):
        m_vec = np.cross(np.stack((results.x,results.y,results.z), axis = 1), np.stack((results.v_x,results.v_y,results.v_z), axis = 1))
        momentumsz.append(np.einsum('ij,ij->i', m_vec, np.full((np.shape(m_vec)), np.array([0, 0, 1]))))
        momentums.append(np.linalg.norm(m_vec,axis=1))

    energyOrbit0 = np.add(selectedPotential[0].energy(np.array([result.x, result.y, result.z])).to_value(),
                          0.5 * np.sum(np.array([result.v_x, result.v_y, result.v_z]).T ** 2, axis=1))
    energyOrbit1 = np.add(selectedPotential[1].energy(np.array([result1.x, result1.y, result1.z])).to_value(),
                          0.5 * np.sum(np.array([result1.v_x, result1.v_y, result1.v_z]).T ** 2, axis=1))
    energyOrbit2 = np.add(selectedPotential[2].energy(np.array([result2.x, result2.y, result2.z])).to_value(),
                          0.5 * np.sum(np.array([result2.v_x, result2.v_y, result2.v_z]).T ** 2, axis=1))

    del (result)
    gc.collect()

    # plot energy and actions
    plt.plot(t1, momentumZ0[0], label="Orbit1")
    plt.plot(t2, momentumZ1[0], label="Orbit2")
    plt.plot(t3, momentumZ2[0], label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[3])
    plt.savefig(selectedFilename[3])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t1, momentumModulo0[0], label="Orbit1")
    plt.plot(t2, momentumModulo1[0], label="Orbit2")
    plt.plot(t3, momentumModulo2[0], label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[4])
    plt.savefig(selectedFilename[4])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t1, energyOrbit0, label="Orbit1")
    plt.plot(t3, energyOrbit2, label="Orbit3")
    plt.plot(t2, energyOrbit1, label="Orbit2")

    plt.legend()
    plt.title(selectedTitle[5])
    plt.savefig(selectedFilename[5])
    plt.show()
    plt.clf()
    gc.collect()

    # plot energy and actions
    plt.plot(t3, relativeVariation(momentumZ2[0].to_value()), label="Orbit3")
    plt.plot(t2, relativeVariation(momentumZ1[0].to_value()), label="Orbit2")
    plt.plot(t1, relativeVariation(momentumZ0[0].to_value()), label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[6])
    plt.savefig(selectedFilename[6])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t3, relativeVariation(momentumModulo2[0].to_value()), label="Orbit3")
    plt.plot(t2, relativeVariation(momentumModulo1[0].to_value()), label="Orbit2")
    plt.plot(t1, relativeVariation(momentumModulo0[0].to_value()), label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[7])
    plt.savefig(selectedFilename[7])
    plt.show()
    plt.clf()
    gc.collect()

    plt.plot(t3, relativeVariation(energyOrbit2), label="Orbit3")
    plt.plot(t2, relativeVariation(energyOrbit1), label="Orbit2")
    plt.plot(t1, relativeVariation(energyOrbit0), label="Orbit1")

    plt.legend()
    plt.title(selectedTitle[8])
    plt.savefig(selectedFilename[8])
    plt.show()
    plt.clf()
    # clear interpreter memory
    gc.collect()