import numpy as np

"""
uLSIFによる確率密度比推定
<ref> 杉山将, 密度比に基づく機械学習の新たなアプローチ, 2009

att:
    sigma -> <float> RBFカーネルの標準偏差
    lam -> <float> 正則化項の重み
    max_center -> <int> RBFカーネルサンプリング点最大数
    x_target -> <np:float:(nt, dim)> ターゲット確率分布よりサンプリングされた実現値
    b -> <int> RBFカーネルサンプリング点数 = min(max_center, nt)
    center -> <np:float:(b, dim)> RBFカーネルサンプル点
    alpha -> <np:float:(b, )> 各基底関数に対する重み

---本クラス利用例---
>model = uLSIF(x_target) #モデルの定義
>model.optim([0.01, 0.1, 1.], [0.1, 1., 10.]) #ハイパーパラメータの決定
>model.fit(x_source) #重み係数の決定
>r = model.pred(x) #密度比の推定
"""
class uLSIF:
    def __init__(self,
                 x_target, #(nt, dim) ターゲット点
                 sigma = None, #RBFカーネルの標準偏差
                 lam = None, #正則化項の重み
                 max_center = 100, #RBFカーネルサンプリング点数
                 ):
        self.sigma = sigma
        self.lam = lam
        self.alpha = None
        self.x_target = np.random.permutation(x_target)

        nt = len(x_target)
        self.b = min(max_center, nt)
        center_index = np.random.choice(nt, self.b, replace = False)
        self.center = self.x_target[center_index]
    
    def norm_X_center(self, x_source):
        x = np.stack([x_source]*self.b, axis = 1)
        return np.sum((x - self.center)**2, axis = -1)
    
    def getH(self, x, sigma):
        norm = self.norm_X_center(x) #(n, b)
        H = np.zeros((self.b, self.b))
        for i in range(self.b):
            for j in range(i, self.b):
                H[i,j] = H[j, i] = np.sum(np.exp(-(norm[:,i] + norm[:,j])/(2.*sigma**2)))/len(x)
        
        return H

    """
    点xにおける密度比の推定値

    input:
        x -> <np:float:(n, dim)> バッチ入力点
    output:
        r -> <n, > 密度比の値
    """
    def pred(self, x):
        assert self.alpha is not None

        norm = self.norm_X_center(x) #(n, b)
        r = np.array([np.sum(self.alpha*np.exp(-n/(2.*self.sigma**2))) for n in norm])
        return r
    
    def geth(self, sigma):
        nt = len(self.x_target)
        norm_target = self.norm_X_center(self.x_target)
        h = np.array([np.sum(np.exp(-norm_target[:,i]/(2.*sigma**2))) / nt for i in range(self.b)])

        return h
    
    """
    ソース確率分布よりサンプリングされた実現値を用いて、密度比回帰モデルのパラメータを決定

    input:
        x_source -> <np:float:(ns, dim)> 実現値
    output: None
        ただしalphaを更新
    """
    def fit(self, x_source):
        assert (self.sigma is not None) and (self.lam is not None)

        H = self.getH(x_source, self.sigma)
        h = self.geth(self.sigma)

        self.alpha = np.linalg.inv(H + self.lam*np.eye(self.b))@h
        self.alpha[self.alpha < 0.] = 0.


    """
    LOOCVに基づき、最適なsigmaおよびlamを決定

    input:
        sigma_list -> <list:float> 検証するsigmaのリスト
        lam_list -> <list:float> 検証するlamのリスト
        x_s -> <np:float:(ns, dim)> ソース確率分布よりサンプリングされた実現値
    output: None
        ただしsigmaとlamが更新される
    """
    def optim(self, sigma_list, lam_list, x_s):
        ns = len(x_s)
        n = min(ns, len(self.x_target))
        x_source = np.random.permutation(x_s)[:n]
        x_target = self.x_target[:n]
        min_sigma = None
        min_lam = None
        min_loocv = float("inf")

        for sigma in sigma_list:
            H = self.getH(x_source, sigma)
            Xs = (np.exp(-self.norm_X_center(x_source)/(2.*sigma**2))).T
            Xt = (np.exp(-self.norm_X_center(x_target)/(2.*sigma**2))).T
            h = np.array([np.mean(xt) for xt in Xt])

            for lam in lam_list:
                B = H + lam*ns/(ns-1)*np.eye(self.b)
                Binv = np.linalg.inv(B)

                B0_diag = h.reshape((1, -1))@Binv@Xs / (ns*np.ones((1, n)) - np.ones((1, self.b))@(Xs*(Binv@Xs)))
                B0 = Binv@(h.reshape((-1,1))@np.ones((1, n))) + Binv@Xs@np.diag(B0_diag[0])

                B1_diag = np.ones((1, self.b))@(Xt*(Binv@Xs)) / (ns*np.ones((1, n)) - np.ones((1, self.b))@(Xs*(Binv@Xs)))
                B1 = Binv@Xt + Binv@Xs@np.diag(B1_diag[0])

                B2 = (ns - 1)/n/(len(self.x_target))*(len(self.x_target)*B0 - B1)
                B2[B2 < 0.] = 0.

                r = np.ones((1, self.b))@(Xs*B2)
                r2 = np.ones((1, self.b))@(Xt*B2)

                loocv = np.sum(r[0]**2)/(2.*n) - np.sum(r2)/n

                if loocv < min_loocv:
                    min_loocv = loocv
                    min_sigma = sigma
                    min_lam = lam
        
        self.sigma = min_sigma
        self.lam = min_lam