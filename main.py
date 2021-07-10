import numpy as np
from matplotlib import pyplot as plt
from KalmanFilter import ContinuousKalmanFilter
from consts import *
from RiccatiSolver import RiccatiSolver


def main():
    def calculate_p(p_prev, Mt, H):
        # tmp1 = 2 * np.matmul(F, self.P_est)  # Ft * Pt + Pt * Ft
        tmp1 = F @ p_prev + p_prev @ F
        tmp2 = p_prev @ H.T / Mt  # Pt * Htt * Mt^-1
        tmp3 = -tmp2 @ H @ p_prev + W_tilde  # - Pt * Htt * Mt^-1 * Ht * Pt + W_tilde

        delta_cov = tmp1 + tmp3  # Ft * Pt + Pt * Ft - Pt * Htt * Mt^-1 * Ht * Pt + W_tilde
        return p_prev + delta_cov*dt  # Ricatti equation for Pt
    'Backpropagation stage - find St values.  m_Q is set to zeros matrix according to the cost function'
    St_list = RiccatiSolver.solve_Riccati_S(m_S0=np.diag([0.5, 0, 0]), m_F=F, v_B=B, m_Q=np.zeros(F.shape), m_R=b/2, t=t_vec[::-1])
    St_list = St_list[::-1]

    'Forward propagation (offline) stage - find state and covariance estimations'
    # Pt_list = [P0]
    # for t in t_vec[:-1]:
    #     H = np.array([[1 / (V * (tf - t)), 0, 0]])  # measurement matrix
    #     Mt = R1 + R2 / ((tf - t) ** 2)
    #     Pt_list.append(calculate_p(Pt_list[-1], Mt, H))

    Pt_list = RiccatiSolver.solve_Riccati_P(m_P0=P0, m_F=F, m_W=W_tilde, t=t_vec[:-1], tf=tf)

    'Forward propagation (online) stage - find state and covariance estimations'
    kalman_filter = ContinuousKalmanFilter(x0, Pt_list, St_list)
    kalman_filter.estimate(plot=True)


if __name__ == '__main__':
    main()