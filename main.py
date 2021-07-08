from matplotlib import pyplot as plt
from KalmanFilter import ContinuousKalmanFilter
from consts import *
from RiccatiSolver import RiccatiSolver

def main():
    # region Backpropagation stage - find St values
    St_list = []
    SOLVER=RiccatiSolver(m_Sf=np.diag([0.5, 0, 0]),m_F=F,v_B = B,m_Q=W_tilde,m_R=b/2,t=t_vec)
    St_list=SOLVER.l_m_S
    # for k in range(len(t_vec)):
    #     St_list.append(np.diag([0.5, 0, 0])) #TODO - update to use RicattiSolver
    #endregion

    #region Forward propagation (offline) stage - find state and covariance estimations
    # Pt_list = []
    # Pt_list,M_list,H_list=a.create_Pt_list(t=t_vec)
    # Kt_list = []
    # for k in range(len(t_vec)):
    #     Pt_list=a.create_Pt_list(t_vec) #TODO - update to use RicattiSolver
    #     Kt_list.append(np.matmul(Pt_list[k], np.matmul(H_list[k],M_list[k])))
    #endregion

    # region Forward propagation (online) stage - find state and covariance estimations
    kalman_filter = ContinuousKalmanFilter(x0, P0, St_list)
    # plt.plot(measurements)
    K_list, x_est_list, P_est_list = kalman_filter.estimate(measurements, show=True)
    #endregion


if __name__ == '__main__':
    main()