import numpy as np

from consts import *
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


class ContinuousKalmanFilter:
    '''
    Kalman filter class.
    Given the system dynamics matrices Ft, B; the measurement matrix H; and the noise matrices W, Mt
    '''
    def __init__(self, x0, Pt_list, St_list):
        self.x_est = x0
        self.St_list = St_list
        self.Pt_list = Pt_list
        self.xtt = 0
        self.t = 0

    def estimate(self, plot=False):
        '''
        Run the full estimation + control algorithm for all time values
        Return the relevant parameters (gain, covariance, state and measurements)
        '''
        def plot_estimates(t, states, gains, covs, measurements):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            fig.suptitle(f'Gains and covariances for 10 steps of the system kalman filter', fontsize=16)

            # subplot 1
            for i, gainVec in enumerate(gains):
                ax1.plot(t[1:], gainVec, '-o', label=f'$x_{i + 1}$ gain')
                ax1.set_title('Kalman gains as a function of time')
                ax1.set_xlabel("Timesteps")
                ax1.set_ylabel("Gain")
            ax1.legend(loc='best')
            ax1.grid(linestyle='dashed')

            # subplot 2
            x1 = [el[0] for el in states]
            x2 = [el[1] for el in states]
            x3 = [el[1] for el in states]

            # plot
            ax2.plot(t, x1, '-o', label='estimated state $x_1$')
            ax2.plot(t, x2, '-o', label='estimated state $x_2$')
            ax2.plot(t, x3, '-o', label='estimated state $x_3$')
            ax2.plot(t[1:], measurements, label='measurements')
            ax2.set_title('Kalman state estimations vs measurements as a function of time')
            ax2.set_xlabel("Timesteps")
            ax2.set_ylabel("x")
            ax2.legend(loc='best')
            ax2.grid(linestyle='dashed')

            p00 = [cov[0][0] for cov in covs]
            p11 = [cov[1][1] for cov in covs]
            p22 = [cov[2][2] for cov in covs]
            ax3.plot(t[1:], p00, '-o', label='estimated covariance p00')
            ax3.plot(t[1:], p11, '-o', label='estimated covariance p11')
            ax3.plot(t[1:], p22, '-o', label='estimated covariance p22')
            ax3.set_title('Covariance estimations as functions of time')
            ax3.set_xlabel("Timesteps")
            ax3.set_ylabel("Covariance")
            ax3.legend(loc='best')
            ax3.grid(linestyle='dashed')

            plt.subplots_adjust(hspace=0.5)
            plt.show()

        gains = []
        states = [self.x_est]  # x0 value
        measurements = []

        for i, t in enumerate(t_vec[:-1]):
            H = np.array([[1 / (V * (tf - self.t)), 0, 0]])  # measurement matrix
            Mt = R1 + R2 / ((tf - self.t) ** 2)

            meas_noise = np.random.normal(0, Mt)
            meas_noise = 0
            meas = H @ states[-1] + meas_noise
            measurements.append(meas[0][0])

            control_cmd = -2/b * B.T @ self.St_list[i] @ states[-1] # optimal controller = -2/b * Bt * S * x_est
            K, x_est = self.estimate_single_iteration(meas, control_cmd, i)

            gains.append(K)
            states.append(x_est)

            self.t = self.t + dt

        gain_y = np.ravel([el[0] for el in gains])
        gain_v = np.ravel([el[1] for el in gains])
        gain_aT = np.ravel([el[2] for el in gains])
        gains = [gain_y, gain_v, gain_aT]

        if plot == True:
            plot_estimates(t_vec, states, gains, self.Pt_list, measurements)

    def estimate_single_iteration(self, measurement, command, iteration):
        '''
        Calculate the Kalman gain using the new measurement
        Estimate the state and the covariance noise
        '''
        'Constants'
        H = np.array([[1 / (V * (tf - self.t)), 0, 0]])  # measurement matrix
        Mt = R1 + R2 / ((tf - self.t) ** 2)

        'time update'
        self.xtt = self.xtt + (H @ self.x_est)[0][0] * dt
        self.P_est = self.Pt_list[iteration]

        'measurement update'
        K = self.P_est @ H.T / Mt # K = Pt * Ht * Mt^-1

        'state estimation'
        self.x_est = (np.matmul(F, self.x_est) + command * B)*dt + np.matmul(K, measurement - self.xtt)

        return K, self.x_est