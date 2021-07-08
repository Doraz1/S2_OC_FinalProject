from matplotlib import pyplot as plt
from KalmanFilter import ContinuousKalmanFilter
from consts import *


def main():
    # region Backpropagation stage - find St values
    St_list = []
    for k in range(len(t_vec)):
        St_list.append(np.diag([0.5, 0, 0])) #TODO - update to use RicattiSolver
    #endregion

    #region Forward propagation (offline) stage - find state and covariance estimations
    # Pt_list = []
    # Kt_list = []
    # for k in range(len(t_vec)):
    #     Pt_list.append(P0) #TODO - update to use RicattiSolver
    #     Kt_list.append(np.matmul(Pt_list[k], np.matmul(H_list[k].T, np.linalg.pinv(M_list[k]))))
    #endregion

    # region Forward propagation (online) stage - find state and covariance estimations
    kalman_filter = ContinuousKalmanFilter(x0, P0, St_list)
    K_list, x_est_list, P_est_list = kalman_filter.estimate(measurements, show=True)
    #endregion


if __name__ == '__main__':
    main()