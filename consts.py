import numpy as np

TAU = 2
Ft = np.array([[0, 1, 0], [0, 0, -1], [0, 0, 1/TAU]])
B = np.array([[0, 1, 0]])
b = 1.52 * 10 ** -2  # no u_k

V = 3000 # average speed
R1 = 15 * 10 ** -6
R2 = 1.67 * 10 ** -3

y0 = 0

P22 = 16 # variance of initial launch velocity gaussian
P33 = 400 # variance of initial launch velocity gaussian

# 1 dimensional noise
v0 = np.random.normal(0, P22)

W = 100 # Qk, target acceleration noise
aT0 = np.random.normal(0, W)

x0 = np.transpose(np.array([[y0, v0, aT0]]))

var_y = 0.0
var_v = P22
var_aT = P33
P0 = np.diag([var_y, var_v, var_aT])# apriori info covariance

dt = 1
tf = 10

measurements = np.array([1, 2, 3, 4, 5, 6])

