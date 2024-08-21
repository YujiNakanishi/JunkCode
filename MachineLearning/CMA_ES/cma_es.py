import numpy as np
import copy
import sys

class CMA_ES:
    generation = 0 #世代数
    def __init__(self,
                 mean : np.ndarray, #初期の平均ベクトル
                 sigma : np.ndarray, #初期の分散値
                 func, #目的関数
                 lam : int = None, #個体数
                 cm : float = 1. #学習率
                 ):
        
        self.mean = copy.deepcopy(mean)
        self.sigma = sigma
        self.func = func

        self.lam = int(4 + np.floor(3*np.log(self.dim))) if lam is None else lam
        self.mu = int(np.floor(self.lam/2.))
        self.cm = cm

        omega_ = np.log((self.lam + 1.)/2.) - np.log(np.arange(self.lam) + 1)
        self.mu_eff = (np.sum(omega_[:self.mu])**2) / np.sum(omega_[:self.mu]**2)

        self.cc = (4. + self.mu_eff/self.dim) / (self.dim + 4. + 2.*self.mu_eff/self.dim)
        self.c1 = 2./((self.dim+1.3)**2 + self.mu_eff)
        cmu_ = (0.5 + 2.*self.mu_eff + 2./self.mu_eff - 4.) / ((self.dim+2)**2 + self.mu_eff)
        self.cmu = min(1. - self.c1, cmu_)
        self.csigma = (self.mu_eff + 2.)/(self.dim + self.mu_eff + 5.)
        self.dsigma = 1. + 2.*max(0., np.sqrt((self.mu_eff - 1.)/(self.dim+1.)) - 1.) + self.csigma

        alphamu = 1. + self.c1 / self.cmu
        alphaeff = 1. + 2.*self.mu_eff/(self.mu_eff + 2.)
        alphapos = (1. - self.c1 - self.cmu)/(self.dim*self.cmu)

        omega_p = np.sum(omega_[omega_ > 0.])
        omega_m = -np.sum(omega_[omega_ < 0.])
        self.omega = np.where(omega_ >= 0., omega_/omega_p, min(alphapos, alphamu, alphaeff)*omega_/omega_m)

        self.pc = np.zeros(self.dim)
        self.psigma = np.zeros(self.dim)
        self.C = np.eye(self.dim)

    @property
    def dim(self):
        ##### 入力変数の次元を返す
        return len(self.mean)
    
    def update(self):
        self.generation += 1
        """サンプリング"""
        y = np.random.multivariate_normal(np.zeros(self.dim), self.C, size = self.lam) #(lam, n)
        x = self.mean + self.sigma*y

        """選択と交叉"""
        values = np.array([self.func(_x) for _x in x])
        min_index = np.argsort(values)
        y_sort = y[min_index]

        y_ = np.sum([self.omega[i]*y_sort[i] for i in range(self.mu)], axis = 0)
        self.mean += self.cm*self.sigma*y_

        """ステップサイズの更新"""
        ##### C^-0.5の計算
        D, B = np.linalg.eigh(self.C)
        #Note: linlg.eigで計算し、かつ世代更新回数が多くなった場合、Cの正定値対称性が崩れ、固有値に複素数が現れた。
        D = np.diag(1./np.sqrt(D))
        sqrt_C_inv = B@D@(B.T)

        self.psigma = (
            (1. - self.cc)*self.psigma
            + np.sqrt(self.csigma*(2. - self.csigma)*self.mu_eff)*(sqrt_C_inv@y_)
        )
        
        """共分散行列の更新"""
        E = np.sqrt(self.dim)*(1. - 1./(4.*self.dim) + 1./(21.*self.dim**2))
        self.sigma *= np.exp(
            self.csigma*self.dsigma
            *(np.linalg.norm(self.psigma)/E -1.)
            )
        
        H = np.linalg.norm(self.pc)/np.sqrt(1. - (1. - self.csigma)**(2*(self.generation+1)))
        h = 1. if H < (1.4 + 2./(self.dim + 1.))*E else 0.
        self.pc = (
            (1. - self.cc)*self.pc
            + h*np.sqrt(self.cc*(2. - self.csigma)*self.mu_eff)*y_
        )
        omega_o = []
        for i in range(self.lam):
            a = 1. if self.omega[i] >= 0. else self.dim/np.sum((sqrt_C_inv@y_sort[i])**2)
            omega_o.append(self.omega[i]*a)
        omega_o = np.array(omega_o)

        delta_h = (1. - h)*self.cc*(2. - self.cc)
        self.C = (
            (1. + self.c1*delta_h - self.c1 - self.cmu*np.sum(self.omega))*self.C
            + self.c1*self.pc.reshape((-1, 1))@self.pc.reshape((1, -1))
        )
        for o_, ys in zip(omega_o, y_sort):
            self.C += self.cmu*o_*ys.reshape((-1, 1))@ys.reshape((1, -1))
        self.C = 0.5*(self.C + self.C.T) + 1e-3*np.eye(len(self.C))
        

def func(x):
    points = np.array([1., 2., 3.])
    a_ans, b_ans, c_ans = 2., 1., -1.
    ans = a_ans*np.sin((2.*points)) + b_ans*np.cos(points) + c_ans
    a, b, c = x
    pred = a*np.sin((2.*points)) + b*np.cos(points) + c

    return float(np.sum((pred - ans)**2))

if __name__ == "__main__":
    mean = np.array([0., 0., 0.])
    sigma = np.array([0.1, 0.1, 0.1])
    model = CMA_ES(mean, sigma, func, lam = 20)
    # print(model.mean)
    for i in range(10):
        model.update()
        print(model.mean)