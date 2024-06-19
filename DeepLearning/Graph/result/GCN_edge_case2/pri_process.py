"""
vtkファイルを読み込み -> データ前処理 -> npyで"./dataset"に保存
計算の重い前処理だけ実行。軽いものはpytorchのDataset側で行う。
"""
import numpy as np
import pyvista as pv
import glob
import os
import sys
import shutil
from scipy import interpolate

from module import priFunctool as pri

pform_dir = "./vtk_output" #p-form解析結果保存場所
vtk_dir = glob.glob(pform_dir + "/*") #pform_dir直下のディレクトリ名のリスト
out_dir = "./dataset" #前処理済みデータの保存場所
phys_names = ["Thickness"] #vtkに格納されているスカラーフィールド等のうち、npyで保存するもののリスト。ex) phys_names = ["Displacement_X"]
scales = ["2mm", "4mm", "10mm"] #解像度のリスト。


for dir in vtk_dir:
    print(dir)
    d = dir.split("_") #ex) d =
    data_dirname = "/{0}_{1}_{2}".format(d[-3], d[-2], d[-1]) # ex) data_name = "m1180_c2_f0.05"
    
    os.mkdir(out_dir + data_dirname) #前処理済みデータの保存場所
    #####念のため元のvtkもコピーしておく
    shutil.copy(dir + "/2mm.vtk", out_dir + data_dirname + "/2mm.vtk")
    shutil.copy(dir + "/4mm.vtk", out_dir + data_dirname + "/4mm.vtk")
    shutil.copy(dir + "/10mm.vtk", out_dir + data_dirname + "/10mm.vtk")

    #####各サイズのvtkファイルを読み込み->pv.UnstructuredGridのインスタンス作成。
    vtk_data_2mm = pv.UnstructuredGrid(pv.read(dir + "/2mm.vtk"))
    vtk_data_4mm = pv.UnstructuredGrid(pv.read(dir + "/4mm.vtk"))
    vtk_data_10mm = pv.UnstructuredGrid(pv.read(dir + "/10mm.vtk"))

    #####隣接行列の保存
    edge_index_2mm = pri.get_AdjacencyMatrix(vtk_data_2mm.cells_dict)
    np.save(out_dir + data_dirname + "/edge_index_2mm", edge_index_2mm)
    edge_index_4mm = pri.get_AdjacencyMatrix(vtk_data_4mm.cells_dict)
    np.save(out_dir + data_dirname + "/edge_index_4mm", edge_index_4mm)
    edge_index_10mm = pri.get_AdjacencyMatrix(vtk_data_10mm.cells_dict)
    np.save(out_dir + data_dirname + "/edge_index_10mm", edge_index_10mm)


    ##### 変形後のノード座標抽出
    vtk_x_2mm = vtk_data_2mm.points
    vtk_x_4mm = vtk_data_4mm.points
    vtk_x_10mm = vtk_data_10mm.points

    #####変形前のノード座標計算
    # p-form解析結果内に、InitialBlank.vtkという変形前のデータがあるが、
    # 変形前後のvtkでNode IDの割り振りが部分的に変わる現象を確認した。
    # 今回は変形後のvtkのNode IDを基準にし、各ノードの変形前の座標はスカラーフィールド"Displacement_XYZ"より計算する。
    displacement_2mm = np.stack((vtk_data_2mm["Displacement_X"], vtk_data_2mm["Displacement_Y"], vtk_data_2mm["Displacement_Z"]), axis = 1)
    displacement_4mm = np.stack((vtk_data_4mm["Displacement_X"], vtk_data_4mm["Displacement_Y"], vtk_data_4mm["Displacement_Z"]), axis = 1)
    displacement_10mm = np.stack((vtk_data_10mm["Displacement_X"], vtk_data_10mm["Displacement_Y"], vtk_data_10mm["Displacement_Z"]), axis = 1)

    vtk_X_2mm = vtk_x_2mm - displacement_2mm #変形前のノード座標値
    np.save(out_dir + data_dirname + "/initX_2mm", vtk_X_2mm)
    vtk_X_4mm = vtk_x_4mm - displacement_4mm
    np.save(out_dir + data_dirname + "/initX_4mm", vtk_X_4mm)
    vtk_X_10mm = vtk_x_10mm - displacement_10mm
    np.save(out_dir + data_dirname + "/initX_10mm", vtk_X_10mm)

    ##########変形後の座標値保存。
    np.save(out_dir + data_dirname + "/x_2mm", vtk_x_2mm)
    np.save(out_dir + data_dirname + "/x_4mm", vtk_x_4mm)
    np.save(out_dir + data_dirname + "/x_10mm", vtk_x_10mm)
    """
    今回、同じ加工条件でも、メッシュ解像度が違えば変形後の加工物の姿勢が異なる。
    入出力に異なる解像度の結果を用いる今回のモデルの場合、深層学習は剛体変位を主に学ぼうとする。
    このような無駄を省くために、高解像度結果の姿勢を是とし、前処理として低解像度結果の位置合わせを行った。具体的には、
    (1)低解像度結果のノード座標分布を高解像度メッシュへ補間
    (2)(1)の結果を更に位置合わせ。
    (3)npyで保存。例えば4mmの結果を2mmの結果に合うように変換した場合、変換結果を"x_interp_2mm_4mm.npy"という名で保存。
    """
    #####補間
    # tips : 今回scipyを利用。RBFInterpolateだと良い仕上がりになった。
    interp_2mm_4mm = interpolate.RBFInterpolator(vtk_X_4mm, vtk_x_4mm)
    x_interp_2mm_4mm = interp_2mm_4mm(vtk_X_2mm)
    interp_2mm_10mm = interpolate.RBFInterpolator(vtk_X_10mm, vtk_x_10mm)
    x_interp_2mm_10mm = interp_2mm_10mm(vtk_X_2mm)
    interp_4mm_10mm = interpolate.RBFInterpolator(vtk_X_10mm, vtk_x_10mm)
    x_interp_4mm_10mm = interp_4mm_10mm(vtk_X_4mm)

    #####位置合わせ
    x_interp_2mm_4mm = pri.shape_registration(x_interp_2mm_4mm, vtk_x_2mm)
    np.save(out_dir + data_dirname + "/x_interp_2mm_4mm", x_interp_2mm_4mm)
    x_interp_2mm_10mm = pri.shape_registration(x_interp_2mm_10mm, vtk_x_2mm)
    np.save(out_dir + data_dirname + "/x_interp_2mm_10mm", x_interp_2mm_10mm)
    x_interp_4mm_10mm = pri.shape_registration(x_interp_4mm_10mm, vtk_x_4mm)
    np.save(out_dir + data_dirname + "/x_interp_4mm_10mm", x_interp_4mm_10mm)

    """
    各ノードの相対的な位置関係を学習するモデルを検討。
    そのため、エッジ特徴量f_ij = x_i - x_j \in R3を計算。
    定義よりf_ij = -f_ijだが、そのような制約はモデルに課していない。
    ノード間の相対位置関係の情報のみからノードの位置座標を求める方法は同フォルダにあるeval.pyを参照。
    """
    dx_2mm = vtk_x_2mm[edge_index_2mm[0]] - vtk_x_2mm[edge_index_2mm[1]] #上記f_ij(2mmの出力用)
    np.save(out_dir + data_dirname + "/dx_2mm", dx_2mm)
    dx_4mm = vtk_x_4mm[edge_index_4mm[0]] - vtk_x_4mm[edge_index_4mm[1]]
    np.save(out_dir + data_dirname + "/dx_4mm", dx_4mm)
    dx_10mm = vtk_x_10mm[edge_index_10mm[0]] - vtk_x_10mm[edge_index_10mm[1]]
    np.save(out_dir + data_dirname + "/dx_10mm", dx_10mm)
    dx_interp_2mm_4mm = x_interp_2mm_4mm[edge_index_2mm[0]] - x_interp_2mm_4mm[edge_index_2mm[1]]
    np.save(out_dir + data_dirname + "/dx_interp_2mm_4mm", dx_interp_2mm_4mm)
    dx_interp_2mm_10mm = x_interp_2mm_10mm[edge_index_2mm[0]] - x_interp_2mm_10mm[edge_index_2mm[1]]
    np.save(out_dir + data_dirname + "/dx_interp_2mm_10mm", dx_interp_2mm_10mm)
    dx_interp_4mm_10mm = x_interp_4mm_10mm[edge_index_4mm[0]] - x_interp_4mm_10mm[edge_index_4mm[1]]
    np.save(out_dir + data_dirname + "/dx_interp_4mm_10mm", dx_interp_4mm_10mm)


    #####フィールドデータの保存
    for phys_name in phys_names:
        p_2mm = vtk_data_2mm[phys_name]
        np.save(out_dir + data_dirname + "/{0}_2mm".format(phys_name), p_2mm)
        p_4mm = vtk_data_4mm[phys_name]
        np.save(out_dir + data_dirname + "/{0}_4mm".format(phys_name), p_4mm)
        p_10mm = vtk_data_10mm[phys_name]
        np.save(out_dir + data_dirname + "/{0}_10mm".format(phys_name), p_10mm)

        interp_2mm_4mm = interpolate.RBFInterpolator(vtk_X_4mm, p_4mm)
        p_interp_2mm_4mm = interp_2mm_4mm(vtk_X_2mm)
        np.save(out_dir + data_dirname + "/{0}_interp_2mm_4mm".format(phys_name), p_interp_2mm_4mm)
        interp_2mm_10mm = interpolate.RBFInterpolator(vtk_X_10mm, p_10mm)
        p_interp_2mm_10mm = interp_2mm_10mm(vtk_X_2mm)
        np.save(out_dir + data_dirname + "/{0}_interp_2mm_10mm".format(phys_name), p_interp_2mm_10mm)
        interp_4mm_10mm = interpolate.RBFInterpolator(vtk_X_10mm, p_10mm)
        p_interp_4mm_10mm = interp_4mm_10mm(vtk_X_4mm)
        np.save(out_dir + data_dirname + "/{0}_interp_4mm_10mm".format(phys_name), p_interp_4mm_10mm)
