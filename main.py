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
    'Backpropagation stage - find St values'
    SOLVER=RiccatiSolver(m_Sf=np.diag([0.5, 0, 0]), m_F=F, v_B = B, m_Q=W_tilde, m_R=b/2, t=t_vec[::-1])
    St_list=SOLVER.l_m_S[::-1]

    'Forward propagation (offline) stage - find state and covariance estimations'
    Pt_list = [P0]
    for t in t_vec[:-1]:
        H = np.array([[1 / (V * (tf - t)), 0, 0]])  # measurement matrix
        Mt = R1 + R2 / ((tf - t) ** 2)
        Pt_list.append(calculate_p(Pt_list[-1], Mt, H))

    'Forward propagation (online) stage - find state and covariance estimations'
    kalman_filter = ContinuousKalmanFilter(x0, Pt_list, St_list)
    kalman_filter.estimate(plot=True)



if __name__ == '__main__':
    main()