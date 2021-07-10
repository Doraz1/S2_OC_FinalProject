import numpy as np
from matplotlib import pyplot as plt
from KalmanFilter import ContinuousKalmanFilter
from consts import *
from RiccatiSolver import RiccatiSolver


def main():
    'Backpropagation stage - find St values.  m_Q is set to zeros matrix according to the cost function'
    St_list = RiccatiSolver.solve_Riccati_S(m_S0=Sf, m_F=F, v_B=B, m_Q=np.zeros(F.shape), m_R=b/2, t=-t_vec)

    'Forward propagation (offline) stage - find state and covariance estimations'
    Pt_list = RiccatiSolver.solve_Riccati_P(m_P0=P0, m_F=F, m_W=W_tilde, t=t_vec[:-1], tf=tf)

    'Forward propagation (online) stage - find state and covariance estimations'
    kalman_filter = ContinuousKalmanFilter(x0, Pt_list, St_list)
    kalman_filter.estimate(plot=True)


if __name__ == '__main__':
    main()