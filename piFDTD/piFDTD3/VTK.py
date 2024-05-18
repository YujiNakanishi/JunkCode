import numpy as np


def writeVTK(filename, shape, size, scalars = [], scalarname = []):
	if len(scalars) != len(scalarname):
		print("Error@piRichard.post.VTK.writeVTK")
		print("len(scalars) should be same with len(scalarname)")
		sys.exit()

	with open(filename, "w") as file:
		#####ジオメトリ構造の書き込み
		file.write("# vtk DataFile Version 2.0\nnumpyVTK\nASCII\n")
		file.write("DATASET STRUCTURED_GRID\n")
		file.write("DIMENSIONS "+str(shape[0])+" "+str(shape[1])+" "+str(shape[2])+"\n")
		file.write("POINTS "+str(shape[0]*shape[1]*shape[2])+" float\n")

		for k in range(shape[2]):
			for j in range(shape[1]):
				for i in range(shape[0]):
					file.write(str(i*size[0])+" "+str(j*size[1])+" "+str(k*size[2])+"\n")

		#####スカラーの書き込み
		if scalars != []:
			file.write("POINT_DATA "+str(shape[0]*shape[1]*shape[2])+"\n")
			
			for _scalar, name in zip(scalars, scalarname):
				#####微小量の丸め込み
				scalar = _scalar.copy()
				scalar[np.abs(scalar) < 1e-20] = 0.
				
				file.write("SCALARS "+name+" float\n")
				file.write("LOOKUP_TABLE default\n")

				for k in range(shape[2]):
					for j in range(shape[1]):
						for i in range(shape[0]):
							file.write(str(scalar[i,j,k])+"\n")