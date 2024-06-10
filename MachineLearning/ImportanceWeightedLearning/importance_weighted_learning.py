"""
重要度重み付き学習によるsinカーブの教師無しドメイン適応

<ref> 松井考太, 熊谷亘, 転移学習, p87
"""


import numpy as np
from ULSIF import uLSIF
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def sampling(num = 50):
    x_seed = np.random.uniform(size = num)
    x_source = []
    x_target = []
    for x in x_seed:
        if np.random.rand() < x:
            x_source.append(x)
        else:
            x_target.append(x)
    
    x_source = np.array(x_source).reshape((-1, 1))
    x_target = np.array(x_target).reshape((-1, 1))

    return x_source, x_target

#####ソース確率分布とターゲット確率分布からxをサンプリング
x_source, x_target = sampling()
ulsif = uLSIF(x_target)
ulsif.optim([0.01, 0.1, 1.], [0.1, 1., 10.], x_source)
ulsif.fit(x_source)

#####各実現値の目的変数を計算
y_source = np.sin(2.*np.pi*x_source[:,0])
y_target = np.sin(2.*np.pi*x_target[:,0])

#####線形回帰モデルを定義。(x_source, y_source)で学習。
model = LinearRegression()
poly_features = PolynomialFeatures(degree = 3)
x_source_poly = poly_features.fit_transform(x_source)
model.fit(x_source_poly, y_source)

#####x_targetに対し予測
x_target_poly = poly_features.fit_transform(x_target)
y_pred = model.predict(x_target_poly)

#####重要度重み付き学習で転移学習
r_source = ulsif.pred(x_source)
model.fit(x_source_poly, y_source, sample_weight = r_source)
y_pred_transfer = model.predict(x_target_poly)

data = np.stack((x_target[:,0], y_target, y_pred, y_pred_transfer), axis = 1)
data = pd.DataFrame(data)
data.to_csv("importance_weighted_learning.csv")