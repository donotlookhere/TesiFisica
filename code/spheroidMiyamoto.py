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

scaleRadius=2*pow(10,-3)
mass=pow(10,5)

def relativeVariation(array):
    return np.divide(np.abs(np.subtract(array,np.full(np.shape(array),array[0]))),np.abs(np.full(np.shape(array),array[0])))

print(agama.getUnits())

points = np.ndarray(shape=(3,3),buffer=np.array([[0.0015,0.0015,0.0010],[0.002,0.002,0.0005],[0.005,0.004,0.003]])).T

print("Points:")
print(*points, sep='\n')

velocity = np.ndarray(shape=(3,3),buffer=np.array([[-7,10,4],[0.2,1,-3],[5,2,-8]]))
starting_points = np.hstack((points.T, velocity))

spheroidalAGAMA = agama.Potential(type="Spheroid",mass=mass,scaleRadius=scaleRadius,axisRatioY=1,axisRatioZ=0.6)
spheroidalExpansionAGAMA = agama.Potential(type="Multipole",density="Spheroid",mass=mass,scaleRadius=scaleRadius,axisRatioY=1,axisRatioZ=0.6)
miyamotoAGAMA= agama.Potential(type="MiyamotoNagai",mass=mass,scaleRadius=scaleRadius,scaleHeight=scaleRadius*(7/10))
miyamotoGALA = gala.potential.MiyamotoNagaiPotential(m=mass,a=scaleRadius,b=scaleRadius*(7/10),units=usys)

potentialsAGAMA=[spheroidalAGAMA,spheroidalExpansionAGAMA,miyamotoAGAMA]
timesAGAMA=[[2,2,2],[2,2,2],[2,2,2]]
trajsizesAGAMA=[[10000000,10000000,10000000],
                [10000000,10000000,10000000],
                [10000000,10000000,10000000]]

titlesAGAMA=[["Spheroidal orbits","Spheroidal orbits","Spheroidal orbits","Spheroidal L_z", "Energy","Spheroidal L_z relative variation","Energy relative variation"],
             ["SpheroidalMultipole orbits","SpheroidalMultipole orbits","SpheroidalMultipole orbits","SpheroidalMultipole L_z", "Energy","SpheroidalMultipole L_z relative variation","Energy relative variation"],
             ["MiyamotoNagai orbits","MiyamotoNagai orbits","MiyamotoNagai orbits","MiyamotoNagai L_z", "Energy","MiyamotoNagai L_z relative variation","Energy relative variation"],]

filenamesAGAMA=[["./output/spheroidMiyamoto/equalTime/SpheroidalOrbits1.png","./output/spheroidMiyamoto/equalTime/SpheroidalOrbits2.png","./output/spheroidMiyamoto/equalTime/SpheroidalOrbits3.png","./output/spheroidMiyamoto/equalTime/SpheroidalL_z.png","./output/spheroidMiyamoto/equalTime/SpheroidalEnergy.png","./output/spheroidMiyamoto/equalTime/SpheroidalL_zRel.png","./output/spheroidMiyamoto/equalTime/SpheroidalEnergyRel.png"],
                ["./output/spheroidMiyamoto/equalTime/SpheroidalMultipoleOrbits1.png","./output/spheroidMiyamoto/equalTime/SpheroidalMultipoleOrbits2.png","./output/spheroidMiyamoto/equalTime/SpheroidalMultipoleOrbits3.png","./output/spheroidMiyamoto/equalTime/SpheroidalMultipoleL_z.png","./output/spheroidMiyamoto/equalTime/SpheroidalMultipoleEnergy.png","./output/spheroidMiyamoto/equalTime/SpheroidalMultipoleL_zRel.png","./output/spheroidMiyamoto/equalTime/SpheroidalMultipoleEnergyRel.png"],
                ["./output/spheroidMiyamoto/equalTime/MiyamotoNagaiOrbits1.png","./output/spheroidMiyamoto/equalTime/MiyamotoNagaiOrbits2.png","./output/spheroidMiyamoto/equalTime/MiyamotoNagaiOrbits3.png","./output/spheroidMiyamoto/equalTime/MiyamotoNagaiL_z.png","./output/spheroidMiyamoto/equalTime/MiyamotoNagaiEnergy.png","./output/spheroidMiyamoto/equalTime/MiyamotoNagaiL_zRel.png","./output/spheroidMiyamoto/equalTime/MiyamotoNagaiEnergyRel.png"]]

for selectedPotential,selectedTime,selectedTrajSize,selectedTitle,selectedFilename in zip(potentialsAGAMA,timesAGAMA,trajsizesAGAMA,titlesAGAMA,filenamesAGAMA):
    result = agama.orbit(potential=selectedPotential, ic=starting_points, time=selectedTime, trajsize=selectedTrajSize,accuracy=1e-16,dtype="float64")

    plt.plot(np.sqrt(np.add(np.square(result[0][1][1:1000000, 0]),np.square(result[0][1][1:1000000, 1]))), result[0][1][1:1000000, 2], label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[0])
    plt.axis('equal')
    plt.savefig(selectedFilename[0])
    plt.show()
    # clear plot
    plt.clf()

    plt.plot(np.sqrt(np.add(np.square(result[1][1][1:1000000:, 0]),np.square(result[1][1][1:1000000:, 1]))), result[1][1][1:1000000:, 2], label="Orbit2")
    plt.legend()
    plt.title(selectedTitle[1])
    plt.axis('equal')
    plt.savefig(selectedFilename[1])
    plt.show()
    # clear plot
    plt.clf()

    plt.plot(np.sqrt(np.add(np.square(result[2][1][:, 0]),np.square(result[2][1][:, 1]))), result[2][1][:, 2], label="Orbit3")
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
    momentumZ0 = []
    momentumZ1 = []
    momentumZ2 = []
    for results, momentumsz in zip([result[0], result[1], result[2]],[momentumZ0, momentumZ1, momentumZ2]):
        m_vec = np.cross(results[1][:, 0:3], results[1][:, 3:6])
        momentumsz.append(np.einsum('ij,ij->i',m_vec, np.full((np.shape(m_vec)),np.array([0, 0, 1]))))

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

    plt.plot(t1, energyOrbit0, label="Orbit1")
    plt.plot(t3, energyOrbit2, label="Orbit3")
    plt.plot(t2, energyOrbit1, label="Orbit2")
    plt.legend()
    plt.title(selectedTitle[4])
    plt.savefig(selectedFilename[4])
    plt.show()
    plt.clf()
    gc.collect()

    # plot energy and actions
    plt.plot(t3, relativeVariation(momentumZ2[0]), label="Orbit3")
    plt.plot(t2, relativeVariation(momentumZ1[0]), label="Orbit2")
    plt.plot(t1, relativeVariation(momentumZ0[0]), label="Orbit1")

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
    plt.title(selectedTitle[6])
    plt.savefig(selectedFilename[6])
    plt.show()
    plt.clf()
    # clear interpreter memory
    gc.collect()

potentialsGALA=[[miyamotoGALA,miyamotoGALA,miyamotoGALA],]
timesGALA=[[2,2,2]]
trajsizesGALA=[[10000000,10000000,10000000]]
integratorGALA=[[gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator,gala.integrate.Ruth4Integrator]]
titlesGALA=[["MiyamotoNagai orbits","MiyamotoNagai orbits","MiyamotoNagai orbits","MiyamotoNagai L_z", "Energy","MiyamotoNagai L_z relative variation","Energy relative variation"]]
filenamesGALA=[["./output/spheroidMiyamoto/symplectic/MiyamotoNagaiOrbits1.png","./output/spheroidMiyamoto/symplectic/MiyamotoNagaiOrbits2.png","./output/spheroidMiyamoto/symplectic/MiyamotoNagaiOrbits3.png","./output/spheroidMiyamoto/symplectic/MiyamotoNagaiL_z.png","./output/spheroidMiyamoto/symplectic/MiyamotoNagaiEnergy.png","./output/spheroidMiyamoto/symplectic/MiyamotoNagaiL_zRel.png","./output/spheroidMiyamoto/symplectic/MiyamotoNagaiEnergyRel.png"]]

for selectedPotential,selectedTime,selectedTrajSize,selectedTitle,selectedFilename,selectedIntegrator in zip(potentialsGALA[0:],timesGALA[0:],trajsizesGALA[0:],titlesGALA[0:],filenamesGALA[0:],integratorGALA[0:]):    #Symplectic integration
    result = selectedPotential[0].integrate_orbit(starting_points[0].T, dt=selectedTime[0]/selectedTrajSize[0], n_steps=selectedTrajSize[0],
                                       Integrator=selectedIntegrator[0],cython_if_possible=True)
    result1 = selectedPotential[1].integrate_orbit(starting_points[1].T, dt=selectedTime[1]/selectedTrajSize[1], n_steps=selectedTrajSize[1],
                                        Integrator=selectedIntegrator[1],cython_if_possible=True)
    result2 = selectedPotential[2].integrate_orbit(starting_points[2].T, dt=selectedTime[2]/selectedTrajSize[2], n_steps=selectedTrajSize[2],
                                        Integrator=selectedIntegrator[2],cython_if_possible=True)

    plt.plot(np.sqrt(np.add(np.square(result.x.to_value()[1:1000000]),np.square(result.y.to_value()[1:1000000]))), result.z.to_value()[1:1000000], label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[0])
    plt.axis('equal')
    plt.show()
    plt.savefig(selectedFilename[0])
    # clear plot
    plt.clf()
    gc.collect()

    plt.plot(np.sqrt(np.add(np.square(result1.x.to_value()[1:1000000]),np.square(result1.y.to_value()[1:1000000]))), result1.z.to_value()[1:1000000], label="Orbit2")
    plt.legend()
    plt.title(selectedTitle[1])
    plt.axis('equal')
    plt.savefig(selectedFilename[1])
    #plt.show()
    # clear plot
    plt.clf()
    gc.collect()

    plt.plot(np.sqrt(np.add(np.square(result2.x.to_value()),np.square(result2.y.to_value()))), result2.z.to_value(), label="Orbit3")
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
    momentumZ0 = []
    momentumZ1 = []
    momentumZ2 = []
    for results, momentumsz in zip([result, result1, result2],
                                              [momentumZ0, momentumZ1, momentumZ2]):
        m_vec = np.cross(np.stack((results.x,results.y,results.z), axis = 1), np.stack((results.v_x,results.v_y,results.v_z), axis = 1))
        momentumsz.append(np.einsum('ij,ij->i', m_vec, np.full((np.shape(m_vec)), np.array([0, 0, 1]))))

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


    plt.plot(t1, energyOrbit0, label="Orbit1")
    plt.plot(t3, energyOrbit2, label="Orbit3")
    plt.plot(t2, energyOrbit1, label="Orbit2")

    plt.legend()
    plt.title(selectedTitle[4])
    plt.savefig(selectedFilename[4])
    plt.show()
    plt.clf()
    gc.collect()

    # plot energy and actions
    plt.plot(t3, relativeVariation(momentumZ2[0].to_value()), label="Orbit3")
    plt.plot(t2, relativeVariation(momentumZ1[0].to_value()), label="Orbit2")
    plt.plot(t1, relativeVariation(momentumZ0[0].to_value()), label="Orbit1")
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
    plt.title(selectedTitle[6])
    plt.savefig(selectedFilename[6])
    plt.show()
    plt.clf()
    # clear interpreter memory
    gc.collect()