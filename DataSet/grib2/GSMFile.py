import sys
import struct
import math
import numpy as np
from inVision.vtk.Structured_Grid import writeVTK

def getSignedInt(file, num):
	b_string = ""
	for i in range(num):
		h2 = file.read(1).hex()
		b_str = ""
		for h in h2:
			b_str += format(int(h, 16), "04b")

		b_string += b_str

	sign = b_string[0]
	value = b_string[1:]

	if sign == "1":
		return -int(value, 2)
	else:
		return int(value, 2)


def checkSection(file):
	section_size = getSignedInt(file, 4)
	section_num = getSignedInt(file, 1)

	return section_num, section_size


def readSection1(file, data):
	string = file.read(12) #1~12.
	data.ref_time = (getSignedInt(file, 2),
				 getSignedInt(file, 1),
				 getSignedInt(file, 1),
				 getSignedInt(file, 1),
				 getSignedInt(file, 1),
				 getSignedInt(file, 1)) #13~19
	string = file.read(2)

def getParamType(file):
	param_cat = getSignedInt(file, 1); param_num = getSignedInt(file, 1)
	param_type = None
	if param_cat == 3:
		if param_num == 1:
			param_type = "Psea"
		elif param_num == 0:
			param_type = "P"
		else:
			param_type = "z"

	elif param_cat == 2:
		if param_num == 2:
			param_type = "U"
		elif param_num == 3:
			param_type = "V"
		else:
			param_type = "W"


	elif param_cat == 0:
		param_type = "T"

	elif param_cat == 1:
		param_type = "H"

	elif param_cat== 6:
		if param_num == 3:
			param_type = "LC"
		elif param_num == 4:
			param_type = "MC"
		elif param_num == 5:
			param_type = "UC"
		else:
			param_type = "TC"




	if param_type is None:
		print("getParamType")
		print(param_cat	)
		print(param_num)
		sys.exit()
	else:
		return param_type


def getSurfaceType(file):
	surface_type = None
	surface_type1 = getSignedInt(file, 1)
	surface_type2 = getSignedInt(file, 1)
	surface_type3 = getSignedInt(file, 4)

	if surface_type1 == 101:
		surface_type = "AveragedSeaLevel"

	elif surface_type1 == 1:
		surface_type = "Ground"

	elif surface_type1 == 103:
		if surface_type3 == 10:
			surface_type = "10m"
		else:
			surface_type = "2m"

	elif surface_type1 == 100:
		surface_type = str(surface_type3)+"hPa"

	if surface_type is None:
		print("getSurfaceType")
		print(surface_type1)
		print(surface_type2)
		print(surface_type3)
		sys.exit()
	else:
		return  surface_type

"""
process : read GRIB2 GSM file
input : filename, data
	filename -> <str> GRIB2 file
	data -> <GSM> GSM class
Note :
- any main process may not call this function explicitly
"""
def readFile(filename, data):
	with open(filename, "rb") as file:
		string_section0 = file.read(16) #1~8.
		readSection1(file, data) # read section 1
		string_section3 = file.read(72)

		while True:
			section_size = getSignedInt(file, 4)
			if section_size == 58:
				#####remove statistical data
				string = file.read(389966)
			elif section_size == 34:
				data.fields.append({})

				#####section 4
				string = file.read(5)
				data.fields[-1]["param_type"] = getParamType(file)
				string = file.read(7)
				data.fields[-1]["time_h"] = getSignedInt(file, 4)
				data.fields[-1]["surface_type"] = getSurfaceType(file)
				string = file.read(6)

				#####section 5
				string = file.read(11)
				data.fields[-1]["R"] = struct.unpack("f", file.read(4)[::-1])[0]
				data.fields[-1]["E"] = getSignedInt(file, 2)
				data.fields[-1]["D"] = getSignedInt(file, 2)
				string = file.read(2)

				######section 6
				string = file.read(6)

				#####section 7
				string = file.read(5)
				bit_line = ""
				for i in range(389880):
					b2 = file.read(1).hex()
					for b in b2:
						bit_line += format(int(b, 16), "04b")

				values = np.zeros((data.grid_num_la,data.grid_num_lo))
				for i in range(data.grid_num_lo):
					for j in range(data.grid_num_la):
						comp_bit = bit_line[12*i*(data.grid_num_la)+12*j : 12*i*(data.grid_num_la)+12*(j+1)]
						comp_val = int(comp_bit, 2)
						values[j, i] = (data.fields[-1]["R"]+comp_val*math.pow(2., data.fields[-1]["E"]))/math.pow(10., data.fields[-1]["D"])


				data.fields[-1]["values"] = values

			else:
				break