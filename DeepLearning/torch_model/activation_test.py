from activation import *

relu_n = ReLU_n()
data = relu_n.test(-2., 2., 100)
data.to_csv("test.csv")