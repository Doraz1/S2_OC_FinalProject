import numpy as np

TAU = 2
F = np.array([[0, 1, 0], [0, 0, -1], [0, 0, -1/TAU]])
B = np.array([[0, 1, 0]]).T
b = 1.52e-2  # no u_k

V = 3000 # average speed
R1 = 15e-6
R2 = 1.67e-3

y0 = 0

P22 = 16 # variance of initial launch velocity gaussian
P33 = 400 # variance of initial launch velocity gaussian

# 1 dimensional noise
v0 = np.random.normal(0, P22)

W = 100 # Qk, target acceleration noise
W_tilde = np.diag([0, 0, W]) # Kalman process noise covariance matrix
aT0 = np.random.normal(0, W)


x0 = np.transpose(np.array([[y0, v0, aT0]]))

var_y = 0.0
var_v = P22
var_aT = P33
P0 = np.diag([var_y, var_v, var_aT])# apriori info covariance

dt = 1
tf = 10
t_vec = np.linspace(0, tf, int(tf/dt)+1)
measurements = np.linspace(1, tf, len(t_vec)-1)

