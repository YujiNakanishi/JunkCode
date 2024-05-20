import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


"""
KANLayer
    ref : "Kolmogorov Arnold Networks", "スプライン関数入門"

Note:
    * 枝切り機能なし

Attribution:
    in_features -> <int> 入力次元
    out_features -> <int> 出力次元
    k -> <int> (k-1)がスプライン関数の次数となる
    base_num -> <int> 基底関数の数
    batch_norm -> <bool>Batch Normを付けるか否か
    x_range -> <torch:float:(in_features, 2)>各説明変数の定義域
    train_knot -> <bool> 節点座標も学習対象に含めるか否か
    knot -> <nn.Param:torch:float:(in_features, base_num + k)> 節点
    b_function -> <nn.Module> 論文中のb(x)
    self.coeffs -> <nn.Param.torch:float:(in_features, base_num, out_features)> スプライン関数の係数
"""
class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, k = 4, base_num = 10, x_range = None, noise_scale = 0.1, batch_norm = False, train_knot = False, b_function = nn.SiLU()):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.base_num = base_num
        self.batch_norm = batch_norm
        self.train_knot = train_knot
        self.b_function = b_function

        #x_rangeはforward計算時に登場する。KANLayer.to("cuda")としたときに、x_rangeもGPUしてもらうために、今回Parameterとして扱った。
        self.x_range = torch.tensor([-1., 1.]).repeat(in_features, 1) if x_range is None else x_range
        self.x_range = nn.Parameter(self.x_range, requires_grad = False)
        
        self.knot, x = self.set_knot(self.x_range)
        self.knot = nn.Parameter(self.knot, requires_grad = train_knot)

        """
        ---スプライン関数の係数の初期設定---
        今回、in_features*out_featuresの数だけスプライン関数を作る。各スプライン関数はbase_numだけの係数を有するため、
        合計in_features*out_features*base_numの係数を扱う。係数self.cを(in_features, base_num, out_features)の配列で管理する。
        """ 
        y = 2.*(torch.rand(in_features, base_num, out_features) - 0.5) * noise_scale / (base_num - 1) #yは補間点xに対するy座標。平均ゼロの乱数で初期設定。
        self.coeffs = self.get_coeffs(x, y, self.knot)
        self.coeffs = nn.Parameter(self.coeffs) #スプライン関数の係数

        """
        ---その他学習パラメータの定義---
        """
        self.lin_b = nn.Linear(in_features, out_features, bias = False) #b(x)のための重みパラメータ
        self.lin_s = nn.Linear(out_features, out_features, bias = False) #スプライン関数のための重みパラメータ
        if batch_norm:
            self.batchnorm1d = nn.BatchNorm1d(out_features, affine = False)
    

    #####論文ではw{b(x)+spline(x)}であるところ、本実装はw1b(x)+w2spline(x)になってる。
    def forward(self, x):
        batch_num = len(x)
        b_output = self.lin_b(self.b_function(x))
        s_output = self.lin_s(self.b_spline_normal(x, self.knot).view(batch_num, -1) @ self.coeffs.view(-1, self.out_features))
        y = b_output + s_output
        
        if self.batch_norm:
            y = self.batchnorm1d(y)
        return y
    
    """
    ---定義域を参考に節点を計算。並びに係数決定に用いる補間点のx座標を計算---
    Input:
        x_range -> <torch:float:(in_features, 2)>各説明変数の定義域
    Output:
        knot -> <torch:tensor:(in_features, base_num+k)> 節点 (等間隔ではない)
        x -> <torch:float:(in_features, base_num)> 各説明変数の補間点x座標 (係数決定に利用する)
    """
    def set_knot(self, x_range, base_num = None):
        if base_num is None:
            base_num = self.base_num

        x_min = x_range[:,0].view(-1, 1); x_max = x_range[:,1].view(-1, 1) #(in_features, 1)
        num_knot_in = base_num - self.k #内部節点の数
        h = ((x_max - x_min) / (base_num - 1)).view(-1, 1) #(in_features, 1) 等間隔に補間点を設置
        a = torch.arange(base_num, device = x_range.device).repeat(self.in_features, 1) #(in_features, base_num)
        x = h*a + x_min #(in_features, base_num)
        #Schoenberg-Whitneyの条件を満たすように、内部節点を決定。
        knot_in = torch.stack([(x[:,i]+x[:,i+self.k])/2. for i in range(num_knot_in)], dim = 1) #(in_features, num_knot_in)
        h_knot = ((knot_in[:,-1] - knot_in[:,0])/(num_knot_in-1)).view(-1, 1) #(in_features, 1) 内部節点の間隔を計算
        b = torch.arange(self.k, device = x_range.device).repeat(self.in_features, 1)
        knot_left = x_min - h_knot*b #付加節点
        knot_right = x_max + h_knot*b #付加節点

        return torch.cat((torch.fliplr(knot_left), knot_in, knot_right), dim = 1), x

    """
    ---正規化された基底関数値を返す---
    Input:
        x -> <batch_size, in_features>
    Output:
        bases -> <batch_size, in_features, base_num>
    """
    def b_spline_normal(self, x, knot):
        x = x.unsqueeze(-1)
        bases = (x > knot[:,:-1])*(x <= knot[:,1:])
        for k in range(1, self.k):
            bases = (x - knot[:,:-(k+1)])/(knot[:,k:-1] - knot[:,:-(k+1)])*bases[:,:,:-1] + \
                    (knot[:,k+1:] - x)/(knot[:,k+1:] - knot[:,1:-k])*bases[:,:,1:]
        
        return bases
    
    """
    ---スプライン補間係数の決定---
    Input:
        X -> <torch:float:(in_features, base_num)> 補間点x座標値
        Y -> <torch:float:(in_features, base_num, out_features)> 補間点y座標値
    Output:
        coeffs -> <torch:float:(in_features, base_num, out_features)> 係数
    """
    def get_coeffs(self, X, Y, knot):
        """
        Input:
            x -> <tensor:float:(base_num)>
            y -> <tensor:float:(base_num)>
        Output:
            c -> <tensor:float:(base_num)>
        """
        def get_coeff(x, y, knot):
            N = len(x)
            A = self.b_spline_normal(x.view(N, 1), knot)[:,0,:] #(base_num, base_num)
            #updateのとき、Aが正則行列になりがちで、連立一次方程式が厳密に解けないことがある。その対策としてpinv。
            A_pinv = torch.linalg.pinv(A)
            return A_pinv @ y

        coeffs = []
        for i in range(self.in_features):
            coeffs.append(torch.stack([get_coeff(X[i], Y[i,:,j], knot) for j in range(self.out_features)], dim = 1))
        coeffs = torch.stack(coeffs, dim = 0)
        return coeffs
    
    """
    サンプルデータからx_rangeを求め、スプライン関数の節点を更新する。
    Input:
        x -> <tensor:float:(batch_num, in_features)> サンプルデータ
    Note:
        安定しない。。。
    """
    def adjust(self, x):
        x_min = torch.min(x, dim = 0)[0] #(in_features, )
        x_max = torch.max(x, dim = 0)[0]
        self.x_range = nn.Parameter(torch.stack((x_min, x_max), dim = 1), requires_grad = False)
        knot, X = self.set_knot(self.x_range) #(in_features, base_num+k), (in_features, base_num)
        self.knot = nn.Parameter(knot, requires_grad = self.train_knot)
        bases = self.b_spline_normal(X.T, self.knot) #(N = base_num, in_features, base_num)
        y = torch.stack([bases[:,i,:]@self.coeffs[i,:,:] for i in range(self.in_features)], dim = 0) #(in_features, base_num, out_features)
        self.coeffs = nn.Parameter(self.get_coeffs(X, y, self.knot))
    
    """
    基底関数の数を増やす
    Input:
        new_base_num -> <int> 新しい基底関数の数
    """
    def add_bases(self, new_base_num):
        x_min = self.x_range[:,0].view(1, -1)
        x_max = self.x_range[:,1].view(1, -1)
        x = x_min + (x_max - x_min)*torch.rand(new_base_num, self.in_features, device = x_min.device) #(new_base_num, in_features)
        bases = self.b_spline_normal(x, self.knot) #(N = base_num, in_features, base_num)
        y = torch.stack([bases[:,i,:]@self.coeffs[i,:,:] for i in range(self.in_features)], dim = 0) #(in_features, new_base_num, out_features) 現時点でのスプライン関数の値。
        
        #(x, y)と一致するようにcoeffを計算
        new_knot, _ = self.set_knot(self.x_range, new_base_num)
        new_coeffs = self.get_coeffs(x.T, y, new_knot)

        self.base_num = new_base_num
        self.knot = nn.Parameter(new_knot, requires_grad = self.train_knot)
        self.coeffs = nn.Parameter(new_coeffs)