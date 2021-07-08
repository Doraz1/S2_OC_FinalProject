import numpy as np

from consts import *
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


class ContinuousKalmanFilter:
    '''
    Kalman filter class.
    Given the system dynamics matrices Ft, B; the measurement matrix H; and the noise matrices W, Mt
    '''
    def __init__(self, x0, P0, St_list):
        self.x_est = x0
        self.P_est = P0
        self.St_list = St_list
        self.xtt = 0
        self.t = 0


    def Iterate_once(self, measurement, command):
        '''
        Estimate the state and the covariance noise
        calculate the Kalman gain using the new measurement
        Update the state and covariance estimations
        '''
        # Initialize constants
        Mt = R1 + R2 / ((tf - self.t)**2)
        H = np.array([[1/(V*(tf - self.t)), 0, 0]])# measurement matrix

        # time update
        self.xtt = self.xtt + np.matmul(H, self.x_est)*dt
        # np.matmul(H, self.x_est) * dt
        self.P_est = self.calculate_p(Mt, H) # Pt = Ft*P + Pt*Ft - Pt*Htt*Mt^-1*Ht*Pt + W_tilde

        # measurement update
        K = np.matmul(self.P_est, H.T) / Mt # K = Pt * Ht * Mt^-1

        self.x_est = (np.matmul(F, self.x_est) + command * B)*dt + np.matmul(K, measurement - self.xtt)
        self.t = self.t + dt

        return K, self.x_est, self.P_est

    def calculate_p(self, Mt, H):
        # tmp1 = 2 * np.matmul(F, self.P_est)  # Ft * Pt + Pt * Ft
        tmp1 = np.matmul(F, self.P_est)+np.matmul( self.P_est,F)
        tmp2 =  np.matmul(self.P_est, np.transpose(H)) /Mt # Pt * Htt * Mt^-1
        tmp3 = -np.matmul(np.matmul(tmp2,H), self.P_est) + W_tilde # - Pt * Htt * Mt^-1 * Ht * Pt + W_tilde

        delta_cov = tmp1 + tmp3 #Ft * Pt + Pt * Ft - Pt * Htt * Mt^-1 * Ht * Pt + W_tilde
        return self.P_est + dt*delta_cov # Ricatti equation for Pt


    # works for a 2x2 covariance matrix
    def covarianceToEllipse(self, cov_matrix, stepNumber, max_c):
        a = cov_matrix[0][0]
        b = cov_matrix[0][1]
        c = cov_matrix[1][1]

        tmp1 = (a + c) / 2
        tmp2 = (a - c) / 2
        radius_vert = tmp1 + np.sqrt(tmp2 ** 2 + b ** 2)
        radius_horiz = tmp1 - np.sqrt(tmp2 ** 2 + b ** 2)
        if b == 0:
            theta = 0 if a >= c else np.pi / 2
        else:
            theta = np.arctan2(radius_vert - a, b)
        cov_ellipse = Ellipse(xy=(stepNumber, int(max_c/2)), width=2 * np.sqrt(radius_vert), height=2 * np.sqrt(radius_horiz),
                         angle=theta * 180 / np.pi)
        return cov_ellipse

    def estimate(self, measurements, show=True):
        def plot_estimates(t, gains, covs):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            fig.suptitle(f'Gains and covariances for 10 steps of the system kalman filter', fontsize=16)

            # subplot 1
            for i, gainVec in enumerate(gains):
                ax1.plot(t[1:], gainVec, label=f'$x_{i + 1}$ gain')
                ax1.set_title('Kalman gains as a function of time')
                ax1.set_xlabel("Timesteps")
                ax1.set_ylabel("Gain")
            ax1.legend(loc='best')

            # subplot 2
            x1 = [el[0] for el in states]
            x2 = [el[1] for el in states]

            # plot
            ax2.plot(t, x1, label='estimated state x1')
            ax2.plot(t, x2, label='estimated state x2')
            ax2.plot(t[1:], measurements, label='measurements')
            ax2.set_title('Kalman state estimations vs measurements as a function of time')
            ax2.set_xlabel("Timesteps")
            ax2.set_ylabel("x")
            ax2.legend(loc='best')

            p00 = [cov[0][0] for cov in covs]
            p11 = [cov[1][1] for cov in covs]
            p22 = [cov[2][2] for cov in covs]
            ax3.plot(t, p00, label='estimated covariance p00')
            ax3.plot(t, p11, label='estimated covariance p11')
            ax3.plot(t, p22, label='estimated covariance p22')
            ax3.set_title('Covariance estimations as functions of time')
            ax3.set_xlabel("Timesteps")
            ax3.set_ylabel("Covariance")
            ax3.legend(loc='best')
            plt.subplots_adjust(hspace=0.5)
            plt.show()

        gains = []
        states = [self.x_est]  # x0 value
        covs = [self.P_est]  # P0 value
        control_cmd = 0

        for i, meas in enumerate(measurements):
            K, x_est, P = self.Iterate_once(meas, control_cmd)
            gains.append(K)
            states.append(x_est)
            covs.append(P)
            control_cmd = -2/b * np.matmul(B.T, np.matmul(self.St_list[i], x_est)) # optimal controller = -2/b * Bt * S * x_est

        t = np.arange(len(measurements) + 1)

        gain_vecs = []
        for i in range(len(gains[0])):
            gains_xi = np.ravel([el[i] for el in gains])
            gain_vecs.append(gains_xi)

        if show:
            plot_estimates(t, gain_vecs, covs)

        return gains, states, covs

