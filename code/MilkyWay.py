import agama
import numpy as np
import matplotlib.pyplot as plt
import gala.potential
import gc
import matplotlib
from gala.units import UnitSystem
import astropy.units as u
import os
from galpy.potential.mwpotentials import  McMillan17
from galpy.util.conversion import get_physical



from astropy.coordinates import SkyCoord
import galpy
from galpy.orbit import Orbit

#matplotlib.use('Agg')

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


McMillan17AGAMA = agama.Potential(os.getenv("HOME")+"/.local/lib/python3.10/site-packages/agama-1.0-py3.10-linux-x86_64.egg/agama/data/McMillan17.ini")

potentialsAGAMA=[McMillan17AGAMA]
timesAGAMA=[[2,2,2]]
trajsizesAGAMA=[[10000000,10000000,10000000]]

titlesAGAMA=[["McMillan17 orbits","McMillan17 orbits","McMillan17 orbits","McMillan17 L_z", "Energy","McMillan17 L_z relative variation","Energy relative variation"],]

filenamesAGAMA=[["./output/McMillan17/equalTime/McMillan17Orbits1.png","./output/McMillan17/equalTime/McMillan17Orbits2.png","./output/McMillan17/equalTime/McMillan17Orbits3.png","./output/McMillan17/equalTime/McMillan17L_z.png","./output/McMillan17/equalTime/McMillan17Energy.png","./output/McMillan17/equalTime/McMillan17L_zRel.png","./output/McMillan17/equalTime/McMillan17EnergyRel.png"]]

# for selectedPotential,selectedTime,selectedTrajSize,selectedTitle,selectedFilename in zip(potentialsAGAMA,timesAGAMA,trajsizesAGAMA,titlesAGAMA,filenamesAGAMA):
#     result = agama.orbit(potential=selectedPotential, ic=starting_points, time=selectedTime, trajsize=selectedTrajSize,accuracy=1e-16,dtype="float64")
#
#     plt.plot(np.sqrt(np.add(np.square(result[0][1][1:1000000, 0]),np.square(result[0][1][1:1000000, 1]))), result[0][1][1:1000000, 2], label="Orbit1")
#     plt.legend()
#     plt.title(selectedTitle[0])
#     plt.axis('equal')
#     plt.savefig(selectedFilename[0])
#     plt.show()
#     # clear plot
#     plt.clf()
#
#     plt.plot(np.sqrt(np.add(np.square(result[1][1][1:1000000:, 0]),np.square(result[1][1][1:1000000:, 1]))), result[1][1][1:1000000:, 2], label="Orbit2")
#     plt.legend()
#     plt.title(selectedTitle[1])
#     plt.axis('equal')
#     plt.savefig(selectedFilename[1])
#     plt.show()
#     # clear plot
#     plt.clf()
#
#     plt.plot(np.sqrt(np.add(np.square(result[2][1][:, 0]),np.square(result[2][1][:, 1]))), result[2][1][:, 2], label="Orbit3")
#     plt.legend()
#     plt.title(selectedTitle[2])
#     plt.axis('equal')
#     plt.savefig(selectedFilename[2])
#     plt.show()
#     # clear plot
#     plt.clf()
#     gc.collect()
#
#     # find actions and energy
#     t1 = result[0][0]
#     t2 = result[1][0]
#     t3 = result[2][0]
#     momentumZ0 = []
#     momentumZ1 = []
#     momentumZ2 = []
#     for results, momentumsz in zip([result[0], result[1], result[2]],[momentumZ0, momentumZ1, momentumZ2]):
#         m_vec = np.cross(results[1][:, 0:3], results[1][:, 3:6])
#         momentumsz.append(np.einsum('ij,ij->i',m_vec, np.full((np.shape(m_vec)),np.array([0, 0, 1]))))
#
#     energyOrbit0 = np.add(selectedPotential.potential(result[0][1][:, 0:3]),
#                           0.5 * np.sum(result[0][1][:, 3:6] ** 2, axis=1))
#     energyOrbit1 = np.add(selectedPotential.potential(result[1][1][:, 0:3]),
#                           0.5 * np.sum(result[1][1][:, 3:6] ** 2, axis=1))
#     energyOrbit2 = np.add(selectedPotential.potential(result[2][1][:, 0:3]),
#                           0.5 * np.sum(result[2][1][:, 3:6] ** 2, axis=1))
#     del(result)
#     gc.collect()
#     # plot energy and actions
#     plt.plot(t1, momentumZ0[0], label="Orbit1")
#     plt.plot(t2, momentumZ1[0], label="Orbit2")
#     plt.plot(t3, momentumZ2[0], label="Orbit3")
#     plt.legend()
#     plt.title(selectedTitle[3])
#     plt.savefig(selectedFilename[3])
#     plt.show()
#     plt.clf()
#     gc.collect()
#
#     plt.plot(t1, energyOrbit0, label="Orbit1")
#     plt.plot(t3, energyOrbit2, label="Orbit3")
#     plt.plot(t2, energyOrbit1, label="Orbit2")
#     plt.legend()
#     plt.title(selectedTitle[4])
#     plt.savefig(selectedFilename[4])
#     plt.show()
#     plt.clf()
#     gc.collect()
#
#     # plot energy and actions
#     plt.plot(t3, relativeVariation(momentumZ2[0]), label="Orbit3")
#     plt.plot(t2, relativeVariation(momentumZ1[0]), label="Orbit2")
#     plt.plot(t1, relativeVariation(momentumZ0[0]), label="Orbit1")
#
#     plt.legend()
#     plt.title(selectedTitle[5])
#     plt.savefig(selectedFilename[5])
#     plt.show()
#     plt.clf()
#     gc.collect()
#
#
#     plt.plot(t3, relativeVariation(energyOrbit2), label="Orbit3")
#     plt.plot(t2, relativeVariation(energyOrbit1), label="Orbit2")
#     plt.plot(t1, relativeVariation(energyOrbit0), label="Orbit1")
#     plt.legend()
#     plt.title(selectedTitle[6])
#     plt.savefig(selectedFilename[6])
#     plt.show()
#     plt.clf()
#     # clear interpreter memory
#     gc.collect()
print(get_physical(McMillan17))
cord = SkyCoord(x=starting_points[:,0]*u.kpc,y=starting_points[:,1]*u.kpc,z=starting_points[:,2]*u.kpc,v_x=starting_points[:,3]*u.km/u.s,v_y=starting_points[:,4]*u.km/u.s,v_z=starting_points[:,5]*u.km/u.s,
                frame='galactocentric',representation_type='cartesian',galcen_distance=np.sqrt(pow(8.21,2)+pow(0.020800000000000003,2))*u.kpc)

gal_int=Orbit(cord,ro=8.21,vo=233.1)

ts=np.linspace(0,2.,10000000)*977.79222168*u.Gyr

gal_int.integrate(ts,McMillan17,method='symplec4_c')

gal_int.plot()