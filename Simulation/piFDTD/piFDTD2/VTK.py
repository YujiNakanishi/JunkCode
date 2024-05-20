"""
/********************/
piRichard.VTK
/********************/
VTKファイル作成のためのクラス、関数を定義。

---ジオメトリについて---
piRichardでは、
・計算領域は六面体
・計算格子は直交格子
である。そのため、VTKファイルのstructureも"STRUCTURED_GRID"に限定している。

---function---
writeVTK
"""
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
		file.write("DIMENSIONS "+str(shape[0])+" "+str(shape[1])+" 1\n")
		file.write("POINTS "+str(shape[0]*shape[1])+" float\n")

		for j in range(shape[1]):
			for i in range(shape[0]):
				file.write(str(i*size)+" "+str(j*size)+" 0\n")

		#####スカラーの書き込み
		if scalars != []:
			file.write("POINT_DATA "+str(shape[0]*shape[1])+"\n")
			
			for _scalar, name in zip(scalars, scalarname):
				#####微小量の丸め込み
				scalar = _scalar.copy()
				scalar[np.abs(scalar) < 1e-20] = 0.
				
				file.write("SCALARS "+name+" float\n")
				file.write("LOOKUP_TABLE default\n")

				for j in range(shape[1]):
					for i in range(shape[0]):
						file.write(str(scalar[i,j])+"\n")