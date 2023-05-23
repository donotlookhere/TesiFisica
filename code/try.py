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
plt.savefig("./output/try/densityCheck.png")
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

potentialsAGAMA=[plummer,BHPotential1,BHPotential2,plummer,BHPotential1,BHPotential2]
timesAGAMA=[[0.05,1,5],[0.05,0.1,5],[0.05,0.1,5],[2,2,2],[2,2,2],[2,2,2]]
trajsizesAGAMA=[[10000000,1000000,10000],
                [10000000,1000000,10000],
                [10000000,1000000,10000],
                [100000000,10000000,100000],
                [10000000,1000000,100000],
                [10000000,1000000,100000]]
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
    result = agama.orbit(potential=selectedPotential, ic=starting_points, time=selectedTime, trajsize=selectedTrajSize)

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

    # find actions and energy
    af = agama.ActionFinder(selectedPotential, False)
    t1 = result[0][0]
    t2 = result[1][0]
    t3 = result[2][0]
    actionsOrbit0 = af(result[0][1])
    actionsOrbit1 = af(result[1][1])
    actionsOrbit2 = af(result[2][1])
    energyOrbit0 = np.add(selectedPotential.potential(result[0][1][:, 0:3]), 0.5 * np.sum(result[0][1][:, 3:6] ** 2, axis=1))
    energyOrbit1 = np.add(selectedPotential.potential(result[1][1][:, 0:3]), 0.5 * np.sum(result[1][1][:, 3:6] ** 2, axis=1))
    energyOrbit2 = np.add(selectedPotential.potential(result[2][1][:, 0:3]), 0.5 * np.sum(result[2][1][:, 3:6] ** 2, axis=1))

    # plot energy and actions
    plt.plot(t1, actionsOrbit0[:, 2], label="Orbit1")
    plt.plot(t2, actionsOrbit1[:, 2], label="Orbit2")
    plt.plot(t3, actionsOrbit2[:, 2], label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[3])
    plt.savefig(selectedFilename[3])
    plt.show()
    plt.clf()

    plt.plot(t1, actionsOrbit0[:, 1] + np.abs(actionsOrbit0[:, 2]), label="Orbit1")
    plt.plot(t2, actionsOrbit1[:, 1] + np.abs(actionsOrbit1[:, 2]), label="Orbit2")
    plt.plot(t3, actionsOrbit2[:, 1] + np.abs(actionsOrbit2[:, 2]), label="Orbit3")
    plt.legend()
    plt.title(selectedTitle[4])
    plt.savefig(selectedFilename[4])
    plt.show()
    plt.clf()

    plt.plot(t1, energyOrbit0, label="Orbit1")
    plt.plot(t3, energyOrbit2, label="Orbit3")
    plt.plot(t2, energyOrbit1, label="Orbit2")
    plt.legend()
    plt.title(selectedTitle[5])
    plt.savefig(selectedFilename[5])
    plt.show()
    plt.clf()

    # plot energy and actions
    plt.plot(t3, relativeVariation(actionsOrbit2[:, 2]), label="Orbit3")
    plt.plot(t2, relativeVariation(actionsOrbit1[:, 2]), label="Orbit2")
    plt.plot(t1, relativeVariation(actionsOrbit0[:, 2]), label="Orbit1")

    plt.legend()
    plt.title(selectedTitle[6])
    plt.savefig(selectedFilename[6])
    plt.show()
    plt.clf()

    plt.plot(t3, relativeVariation(actionsOrbit2[:, 1] + np.abs(actionsOrbit2[:, 2])), label="Orbit3")
    plt.plot(t2, relativeVariation(actionsOrbit1[:, 1] + np.abs(actionsOrbit1[:, 2])), label="Orbit2")
    plt.plot(t1, relativeVariation(actionsOrbit0[:, 1] + np.abs(actionsOrbit0[:, 2])), label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[7])
    plt.savefig(selectedFilename[7])
    plt.show()
    plt.clf()

    plt.plot(t3, relativeVariation(energyOrbit2), label="Orbit3")
    plt.plot(t2, relativeVariation(energyOrbit1), label="Orbit2")
    plt.plot(t1, relativeVariation(energyOrbit0), label="Orbit1")
    plt.legend()
    plt.title(selectedTitle[8])
    plt.savefig(selectedFilename[8])
    plt.show()
    plt.clf()
    # clear interpreter memory
    del (result)
    gc.collect()


# # ########
# potentialsGALA=[[Hernquist,Hernquist,Hernquist],
#                 [Hernquist,Hernquist,Hernquist]]
# timeintervalsGALA=[[0.00000001,0.000001,0.0000005],
#                    [0.000001,0.000001,0.000001]]
# nstepsGALA=[[10000,10000,10000],
#             [10000,10000,10000]]
# integratorGALA=[[gala.integrate.DOPRI853Integrator,gala.integrate.DOPRI853Integrator,gala.integrate.DOPRI853Integrator],
#                 [gala.integrate.DOPRI853Integrator,gala.integrate.DOPRI853Integrator,gala.integrate.DOPRI853Integrator]]
# titlesGALA=[["Hernquist orbits","Hernquist orbits","Hernquist orbits","Hernquist L_z","Total angular momentum L","Energy","Hernquist L_z relative variation","Total angular momentum L relative variation","Energy relative variation"],
#             ["Hernquist orbits","Hernquist orbits","Hernquist orbits","Hernquist L_z","Total angular momentum L","Energy","Hernquist L_z relative variation","Total angular momentum L relative variation","Energy relative variation"]]
# filenamesGALA=[["./output/try/differentTime/HernquistOrbits1.png","./output/try/differentTime/HernquistOrbits2.png","./output/try/differentTime/HernquistOrbits3.png","./output/try/differentTime/HernquistL_z.png","./output/try/differentTime/Hernquist_angular_momentum.png","./output/try/differentTime/HernquistEnergy.png","./output/try/differentTime/HernquistL_zRel.png","./output/try/differentTime/Hernquist_angular_momentumRel.png","./output/try/differentTime/HernquistEnergyRel.png"],
#                ["./output/try/equalTime/HernquistOrbits1.png","./output/try/equalTime/HernquistOrbits2.png","./output/try/equalTime/HernquistOrbits3.png","./output/try/equalTime/HernquistL_z.png","./output/try/equalTime/Hernquist_angular_momentum.png","./output/try/equalTime/HernquistEnergy.png","./output/try/equalTime/HernquistL_zRel.png","./output/try/equalTime/Hernquist_angular_momentumRel.png","./output/try/equalTime/HernquistEnergyRel.png"]]
#
#
# for selectedPotential,selectedDt,selectedNsteps,selectedIntegrator,selectedTitle,selectedFilename in zip(potentialsGALA,timeintervalsGALA,nstepsGALA,integratorGALA,titlesGALA,filenamesGALA):
#     #Hernquist integration
#     result = selectedPotential[0].integrate_orbit(starting_points[0].T, dt=selectedDt[0], n_steps=selectedNsteps[0],
#                                        Integrator=selectedIntegrator[0])
#     result1 = selectedPotential[1].integrate_orbit(starting_points[1].T, dt=selectedDt[1], n_steps=selectedNsteps[1],
#                                         Integrator=selectedIntegrator[1])
#     result2 = selectedPotential[2].integrate_orbit(starting_points[2].T, dt=selectedDt[2], n_steps=selectedNsteps[2],
#                                         Integrator=selectedIntegrator[2])
#
#     plt.plot(result.x, result.y, label="Orbit1")
#     plt.legend()
#     plt.title(selectedTitle[0])
#     plt.axis('equal')
#     plt.savefig(selectedFilename[0])
#     plt.show()
#     # clear plot
#     plt.clf()
#
#     plt.plot(result1.x, result1.y, label="Orbit2")
#     plt.legend()
#     plt.title(selectedTitle[1])
#     plt.axis('equal')
#     plt.savefig(selectedFilename[1])
#     plt.show()
#     # clear plot
#     plt.clf()
#
#     plt.plot(result2.x, result2.y, label="Orbit3")
#     plt.legend()
#     plt.title(selectedTitle[2])
#     plt.axis('equal')
#     plt.savefig(selectedFilename[2])
#     plt.show()
#     # clear plot
#     plt.clf()
#
#     # find actions and energy
#     t1 = result.t
#     t2 = result1.t
#     t3 = result2.t
#     momentumModulo0 = []
#     momentumModulo1 = []
#     momentumModulo2 = []
#     momentumZ0 = []
#     momentumZ1 = []
#     momentumZ2 = []
#     for results, momentums, momentumsz in zip([result, result1, result2],
#                                               [momentumModulo0, momentumModulo1, momentumModulo2],
#                                               [momentumZ0, momentumZ1, momentumZ2]):
#         for x, y, z, v_x, v_y, v_z in zip(results.x, results.y, results.z, results.v_x, results.v_y, results.v_z):
#             m_vec = np.cross(np.array([x, y, z]), np.array([v_x, v_y, v_z]))
#             momentumsz.append(np.linalg.norm(np.dot(m_vec, np.array([0, 0, 1]))))
#             momentums.append(np.linalg.norm(m_vec))
#
#     energyOrbit0 = np.add(selectedPotential[0].potential(np.array([result.x, result.y, result.z]).T),
#                           0.5 * np.sum(np.array([result.v_x, result.v_y, result.v_z]).T ** 2, axis=1))
#     energyOrbit1 = np.add(selectedPotential[1].potential(np.array([result1.x, result1.y, result1.z]).T),
#                           0.5 * np.sum(np.array([result1.v_x, result1.v_y, result1.v_z]).T ** 2, axis=1))
#     energyOrbit2 = np.add(selectedPotential[2].potential(np.array([result2.x, result2.y, result2.z]).T),
#                           0.5 * np.sum(np.array([result2.v_x, result2.v_y, result2.v_z]).T ** 2, axis=1))
#
#     # plot energy and actions
#     plt.plot(t1, momentumZ0, label="Orbit1")
#     plt.plot(t2, momentumZ1, label="Orbit2")
#     plt.plot(t3, momentumZ2, label="Orbit3")
#     plt.legend()
#     plt.title(selectedTitle[3])
#     plt.savefig(selectedFilename[3])
#     plt.show()
#     plt.clf()
#
#     plt.plot(t1, momentumModulo0, label="Orbit1")
#     plt.plot(t2, momentumModulo1, label="Orbit2")
#     plt.plot(t3, momentumModulo2, label="Orbit3")
#     plt.legend()
#     plt.title(selectedTitle[4])
#     plt.savefig(selectedFilename[4])
#     plt.show()
#     plt.clf()
#
#     plt.plot(t1, energyOrbit0, label="Orbit1")
#     plt.plot(t3, energyOrbit2, label="Orbit3")
#     plt.plot(t2, energyOrbit1, label="Orbit2")
#
#     plt.legend()
#     plt.title(selectedTitle[5])
#     plt.savefig(selectedFilename[5])
#     plt.show()
#     plt.clf()
#
#     # plot energy and actions
#     plt.plot(t3, relativeVariation(momentumZ2), label="Orbit3")
#     plt.plot(t2, relativeVariation(momentumZ1), label="Orbit2")
#     plt.plot(t1, relativeVariation(momentumZ0), label="Orbit1")
#     plt.legend()
#     plt.title(selectedTitle[6])
#     plt.savefig(selectedFilename[6])
#     plt.show()
#     plt.clf()
#
#     plt.plot(t3, relativeVariation(momentumModulo2), label="Orbit3")
#     plt.plot(t2, relativeVariation(momentumModulo1), label="Orbit2")
#     plt.plot(t1, relativeVariation(momentumModulo0), label="Orbit1")
#     plt.legend()
#     plt.title(selectedTitle[7])
#     plt.savefig(selectedFilename[7])
#     plt.show()
#     plt.clf()
#
#     plt.plot(t3, relativeVariation(energyOrbit2), label="Orbit3")
#     plt.plot(t2, relativeVariation(energyOrbit1), label="Orbit2")
#     plt.plot(t1, relativeVariation(energyOrbit0), label="Orbit1")
#
#     plt.legend()
#     plt.title(selectedTitle[8])
#     plt.savefig(selectedFilename[8])
#     plt.show()
#     plt.clf()
#     # clear interpreter memory
#     del (result)
#     gc.collect()