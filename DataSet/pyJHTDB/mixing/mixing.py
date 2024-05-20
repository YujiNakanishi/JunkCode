import numpy as np
import pyJHTDB

x_min = 0.
y_min = 0.
z_min = 0.
x_max = 2.*np.pi
y_max = 2.*np.pi
z_max = 2.*np.pi
dt = 0.04
time_step = 1015
spatialInterp = 6 #6 point Lagrange
temporalInterp = 0 #no time interpolation
FD4Lag4 = 44 #4 point Lagrange interp for derivatives
length = 1024

x = (np.arange(length)/length)*x_max + x_min
y = (np.arange(length)/length)*y_max + y_min

xx, yy = np.meshgrid(x, y)
xy = (np.stack((xx, yy), axis = -1)).reshape((-1, 2))
points = np.concatenate((xy, np.ones((len(xy), 1))), axis = 1)

lTDB = pyJHTDB.libJHTDB()
lTDB.initialize()
auth_token = "edu.jhu.pha.turbulence.testing-201311"
lTDB.add_token(auth_token)

stride = int((length*length)/4096)
for ts in range(time_step):
    results = []
    print(ts/time_step)
    for s in range(stride):
        point = points[4096*s:4096*(s+1)]
        result = lTDB.getData(dt*ts, point.astype(np.float32), data_set = 'mixing', sinterp = spatialInterp, tinterp = temporalInterp, getFunction = 'getVelocity')
        results.append(result)
    

    results = np.stack(results, axis = 0)
    results = results.reshape((length, length, 3))
    np.save("./mixing/velocity"+str(ts).zfill(5)+".npy", results)