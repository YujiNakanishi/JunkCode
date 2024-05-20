"""
sampling from channel flow data
"""

import numpy as np
import pyJHTDB
import pyvista as pv


T = pyJHTDB.dbinfo.channel["time"][-1] #max time[s] of this dataset
time = np.random.rand()*T #sample a time (continuous value is possible)
spatialInterp = 6 #6 point Lagrange
temporalInterp = 0 #no time interpolation
FD4Lag4 = 44 #4 point Lagrange interp for derivatives

x = np.arange(0., 8.*np.pi, 0.5)
y = np.arange(-1., 1., 0.5)
z = np.arange(0., 3.*np.pi, 0.5)

geo = pv.RectilinearGrid(x, y, z)

lTDB = pyJHTDB.libJHTDB()
lTDB.initialize()

auth_token = "edu.jhu.pha.turbulence.testing-201311"
lTDB.add_token(auth_token)
result = lTDB.getData(time, geo.points.astype(np.float32), data_set = 'channel', sinterp = spatialInterp, tinterp = temporalInterp, getFunction = 'getVelocity')
E = np.sum(result**2, axis = 1)
geo.point_data["pressure"] = result
geo.save("test.vtk")