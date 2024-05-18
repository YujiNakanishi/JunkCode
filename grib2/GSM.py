from pigrib2 import GSMFile
from inVision.vtk.Structured_Grid import writeVTK
import numpy as np


"""
/**************/
GSM : GSM decoder
/**************/
---field---
ref_time -> <tuple of int> (year, month, day, hour, minute, second)
radius -> <float> radius of the earth [m] = 6371000.
grid_num_la -> <int> number of grid around latitude
grid_num_lo -> <int> number of grid around longitude
la1 -> <float> latitude of initial grid
lo1 -> <float> longitude of initial grid
la2 -> <float> latitude of last grid
lo2 -> <float> longitude of last grid
fields -> <list of dictionary>
	"param_type" -> <str> expression of physics value
		"Psea" : Sea Level Correction Barometric Pressure
		"P" : Pressure 
				
	"time_h" -> <int> forecast time
	
	"surafce_type" -> <str> expression of surface
		"AveragedSeaLevel" : Averaged Sea Level
	
	"R", "E", "D" -> <num>
"""
class GSM:
	"""
	input : filename -> <str> GRIB2 file
	"""
	def __init__(self, filename):
		self.radius = 6371000.
		self.grid_num_la = 720
		self.grid_num_lo = 361
		self.shape = (self.grid_num_la, self.grid_num_lo)
		self.la1 = 90.; self.lo1 = 0.; self.la2 = -90.; self.lo2 = 359.5
		self.fields = []
		GSMFile.readFile(filename, self)

	def __call__(self, param_type, surface_type):
		for field in self.fields:
			if (field["param_type"] == param_type) & (field["surface_type"] == surface_type):
				return field["values"]

		return None

	def createVTK(self, param_types, surface_types, filename):
		values = []
		names = [param_type+"/"+surface_type for param_type, surface_type in zip(param_types, surface_types)]

		for param_type, surface_type in zip(param_types, surface_types):
			value = self(param_type, surface_type)
			value = np.flipud(value.T).copy()
			values.append(value.T)

		writeVTK(filename, (self.grid_num_la, self.grid_num_lo), (1., 1.), values, names)