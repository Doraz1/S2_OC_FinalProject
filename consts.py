import numpy as np

'Settings'
# np.random.seed(0)
# np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)

'Constants'
TAU = 2
dt = 0.1
tf = 10
t_vec = np.linspace(0, tf, int(tf/dt)+1)
F = np.array([[0, 1, 0], [0, 0, -1], [0, 0, -1/TAU]])
B = np.array([[0, 1, 0]]).T
b = 1.52e-2  # no u_k
V = 3000 # average speed
R1 = 15e-6
R2 = 1.67e-3
W = 100 # Qk, target acceleration noise
G = np.array([[0, 0, 1]]).T
W_tilde = np.diag([0, 0, W]) # Kalman process noise covariance matrix

'State values and variances'
var_v = P22 = 16 # variance of initial launch velocity
var_aT = P33 = 400 # variance of initial target acceleration
var_y = 0.0
P0 = np.diag([var_y, var_v, var_aT])# apriori info covariance
y0 = 0
v0 = np.random.normal(0, np.sqrt(P22))
aT0 = np.random.normal(0, var_aT)

# x0 = np.transpose(np.array([[y0, v0, aT0]]))
x0 = np.transpose(np.array([[0, v0, 0]]))

'Generate target path'