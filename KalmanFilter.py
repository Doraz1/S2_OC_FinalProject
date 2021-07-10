import numpy as np

from consts import *
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from RiccatiSolver import RiccatiSolver
from scipy.integrate import odeint


class ContinuousKalmanFilter:
    '''
    Kalman filter class.
    Given the system dynamics matrices Ft, B; the measurement matrix H; and the noise matrices W, Mt
    '''
    def __init__(self, x0, Pt_list, St_list):
        self.x_gt = x0
        self.x_est = x0 * 0
        self.St_list = St_list
        self.Pt_list = Pt_list
        self.xtt = 0
        self.t = 0

    def estimate(self, plot=False):
        '''
        Run the full estimation + control algorithm for all time values
        Return the relevant parameters (gain, covariance, state and measurements)
        '''
        gains = np.zeros((3, 1))
        states = self.x_est  # x0 value
        gt_states = self.x_gt  # x0 value
        measurements = []
        commands = []

        for i, t in enumerate(t_vec[:-1]):
            H = np.array([[1 / (V * (tf - self.t)), 0, 0]])  # measurement matrix
            Mt = R1 + R2 / ((tf - self.t) ** 2)

            control_cmd = -2/b * B.T @ self.St_list[i] @ states[:, -1] # optimal controller = -2/b * Bt * S * x_est
            K, x_est, x_gt, measurement = self.estimate_single_iteration(control_cmd, i)

            gains = np.hstack((gains, K))
            measurements.append(measurement)
            commands = np.hstack((commands, control_cmd))
            states = np.hstack((states, x_est))
            gt_states = np.hstack((gt_states, x_gt))

            self.t = self.t + dt

        gains = gains[:, 1:]
        if plot == True:
            ContinuousKalmanFilter.plot_gains(t_vec, gains)
            ContinuousKalmanFilter.plot_states(t_vec, states, gt_states, commands)
            ContinuousKalmanFilter.plot_P(t_vec, self.Pt_list)
            plt.show()

    def estimate_single_iteration(self, command, iteration):
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

        'Ground true'
        self.x_gt += (np.matmul(F, self.x_gt) + command * B) * dt + G * np.random.normal(0, np.sqrt(W))*dt
        measurement = H @ self.x_gt + np.random.normal(0, np.sqrt(Mt))

        'state estimation'
        self.x_est += (np.matmul(F, self.x_est) + command * B)*dt + np.matmul(K, measurement - H @ self.x_est)*dt

        return K, self.x_est, self.x_gt, measurement

    @staticmethod
    def plot_gains(t, gains):
        plt.figure(1)
        for i, row in enumerate(gains):
            plt.subplot(3, 1, i + 1)
            plt.plot(t[1:], gains[i, :], '-o')
            plt.ylabel('$K_{}$'.format(i))
            plt.xlabel('time [s]')
            plt.grid(linestyle='dashed')

        plt.subplots_adjust(hspace=0.5)
        # plt.show()

    @staticmethod
    def plot_states(t, states, gt_states, commands):
        plt.figure(2)
        for i, row in enumerate(states):
            plt.subplot(4, 1, i + 1)
            plt.plot(t, states[i, :], '-o')
            plt.plot(t, gt_states[i, :], '-o')
            plt.legend(['estimate state', 'ground true'])
            plt.ylabel('$x_{}$'.format(i))
            plt.xlabel('time [s]')
            plt.grid(linestyle='dashed')

        plt.subplot(4, 1, 4)
        plt.plot(t[1:], commands - states[2, :-1], '-o')
        plt.plot(t[1:], commands - gt_states[2, :-1], '-o')
        plt.legend(['$a_P - a_{T estimate}$', '$a_P - a_{T}$'])
        plt.ylabel('$a_P - a_T$')
        plt.xlabel('time [s]')
        plt.grid(linestyle='dashed')

        plt.subplots_adjust(hspace=0.5)
        # plt.show()

    @staticmethod
    def plot_P(t, l_P):
        p00 = [cov[0][0] for cov in l_P]
        p11 = [cov[1][1] for cov in l_P]
        p22 = [cov[2][2] for cov in l_P]
        tmp = [p00, p11, p22]

        plt.figure(3)
        for i, row in enumerate(tmp):
            plt.subplot(3, 1, i + 1)
            plt.plot(t[1:], row, '-o')
            plt.ylabel('$P_{}$'.format(i))
            plt.xlabel('time [s]')
            plt.grid(linestyle='dashed')

        plt.subplots_adjust(hspace=0.5)
        # plt.show()