import glob
import random
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

"""
Input:
    data_dir -> <str> 前処理済みデータセットのパス。ex) pri_process.pyの出力場所
    train_ratio -> <float> 訓練データ比率
Output:
    <list:dict:fem_data> fem_data = {"LR_Scale" : int, "HR_scale" : int, "file_dir" : str}

Note:
torch.Dataset(もしくはtorch_geometricのDataset)のi番目のデータが呼び出されたとき、
fem_data[i]["file_dir"]にある、"LR_Scale"mm("HR_Scale"mm)を入力(出力)として返す。
"""
def train_test_split(data_dir = "./dataset", train_ratio = 0.8):
    list_dir = glob.glob(data_dir+"/*")

    data_list = []
    for dir in list_dir:
        data_list.append({"LR_scale" : 10, "HR_scale" : 4, "file_dir" : dir})
        data_list.append({"LR_scale" : 10, "HR_scale" : 2, "file_dir" : dir})
        data_list.append({"LR_scale" : 4, "HR_scale" : 2, "file_dir" : dir})
    
    random.shuffle(data_list)

    train_num = int(train_ratio * len(data_list))
    train_list = data_list[:train_num]
    test_list = data_list[train_num:]

    return train_list, test_list



class Abst_Dataset(Dataset):
    """
    data_list -> <list:fem_data>
    """
    def __init__(self, data_list, device = "cuda"):
        super().__init__()
        self.data_list = data_list
        self.device = device
    
    def len(self):
        return len(self.data_list)
    
    ##### idx番目のデータを出力
    def get(self, idx):
        fem_data = self.data_list[idx]
        return self.define_Data(fem_data)
    
    ##### 「npyファイルの読み込み -> 諸々の処理 -> tensorへの変換」を行う。
    # もしもデータの入出力がノード座標なら、npyデータ = ノード座標値ゆえ"諸々の処理"はほとんど不要。
    def define_Data(self, fem_data):
        raise NotImplementedError


"""
入出力に変形後のノード座標を使用。
ただし、追加の前処理として単位を mmからcmに変換している。
"""
class Coord(Abst_Dataset):
    def define_Data(self, fem_data):
        lr_scale = fem_data["LR_scale"]
        hr_scale = fem_data["HR_scale"]

        x = np.load(fem_data["file_dir"] + "/x_interp_{0}mm_{1}mm.npy".format(hr_scale, lr_scale)) #入力
        x /= 10. #単位変換

        y = np.load(fem_data["file_dir"] + "/x_{}mm.npy".format(hr_scale)) #出力
        y /= 10.

        edge_index = np.load(fem_data["file_dir"] + "/edge_index_{}mm.npy".format(hr_scale)) #隣接行列

        x = torch.tensor(x, dtype = torch.float32, device = self.device)
        y = torch.tensor(y, dtype = torch.float32, device = self.device)
        edge_index = torch.tensor(edge_index, dtype = torch.long, device = self.device)

        data = Data(x = x, y = y, edge_index = edge_index)
        return data


"""
入出力として変形後のノード間相対位置関係をエッジ特徴量に与える。ノード特徴量はThickness。
追加の前処理として単位を mmからcmに変換している。
Thicknessを選んだ根拠はなんとなく。
ノード特徴量ではなくエッジ特徴量から学習する。
"""
class RelDistance(Abst_Dataset):
    def define_Data(self, fem_data):
        lr_scale = fem_data["LR_scale"]
        hr_scale = fem_data["HR_scale"]

        dx = np.load(fem_data["file_dir"] + "/dx_interp_{0}mm_{1}mm.npy".format(hr_scale, lr_scale)) / 10. #低解像度結果におけるノード間相対位置(cmに単位変換)
        dy = np.load(fem_data["file_dir"] + "/dx_{}mm.npy".format(hr_scale)) / 10. #高解像度結果におけるノード間相対位置(cmに単位変換)
        edge_index = np.load(fem_data["file_dir"] + "/edge_index_{}mm.npy".format(hr_scale)) #隣接行列

        #####ノード特徴量としてThicknessの変化量 [mm]を利用。変形前の板厚はすべて1mmだと認識している。
        x = np.load(fem_data["file_dir"] + "/Thickness_interp_{0}mm_{1}mm.npy".format(hr_scale, lr_scale)) - 1.
        x = torch.tensor(x.reshape((-1, 1)), dtype = torch.float32, device = self.device)
        dx = torch.tensor(dx, dtype = torch.float32, device = self.device)
        dy = torch.tensor(dy, dtype = torch.float32, device = self.device)
        edge_index = torch.tensor(edge_index, dtype = torch.long, device = self.device)

        data = Data(x = x, edge_index = edge_index, edge_attr = dx, y = dy)
        return data