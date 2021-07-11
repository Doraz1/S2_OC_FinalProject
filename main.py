import random

import numpy as np
from matplotlib import pyplot as plt
from KalmanFilter import ContinuousKalmanFilter
from consts import *
from RiccatiSolver import RiccatiSolver


def solve_Riccati_K(m_P):
    K = []
    index_t=0
    while index_t<100:
        t=t_vec[index_t]
        H = np.array([[1 / (V * (tf - t)), 0, 0]])  # measurement matrix
        Mt = R1 + R2 / ((tf - t) ** 2)
        k = m_P[index_t] @ H.T / Mt
        K.append(k)
        index_t=index_t+1
    return K


def main(gaincheck):
    'Backpropagation stage - find St values.  m_Q is set to zeros matrix according to the cost function'
    St_list = RiccatiSolver.solve_Riccati_S(m_S0=np.diag([0.5, 0, 0]), m_F=F, v_B=B, m_Q=np.zeros(F.shape), m_R=b/2, t=-t_vec)
    St_list = St_list[::-1]
    'Forward propagation (offline) stage - find state and covariance estimations'
    Pt_list = RiccatiSolver.solve_Riccati_P(m_P0=P0, m_F=F, m_W=W_tilde, t=t_vec[:-1], tf=tf)

    K_list= solve_Riccati_K(Pt_list)
    'Forward propagation (online) stage - find state and covariance estimations'
    kalman_filter = ContinuousKalmanFilter(x0, Pt_list, St_list,K_list)
    # l=0
    # j=np.zeros(3)
    j=kalman_filter.estimate(plot=False,gaincheck=gaincheck)
    return j
if __name__ == '__main__':

    j=0
    j1=[]
    j2=[]
    j3=[]
    gaincheck = [1, 1.01, 0.99]
    index_t = 0

    for i in range(50):
        random.seed(i)
        for gc  in gaincheck:
            j=main(gaincheck=gc)
            if index_t==0:
                j1.append(j)
            if index_t == 1:
                j2.append(j)
            if index_t == 2:
                j3.append(j)
            index_t=index_t+1
        index_t=0


    fig, ax = plt.subplots()
    ax.plot(j1,'o',label='optimal')
    ax.plot(j2,'o',label='optimal*1.01')
    ax.plot(j3,'o',label='optimal*0.99')
    legend = ax.legend(loc='upper center', shadow=False, fontsize='x-large')
    legend.get_frame().set_facecolor('C0')
    plt.show()