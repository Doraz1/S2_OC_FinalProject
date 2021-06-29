from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from tabulate import tabulate
from consts import *


class ContinuousKalmanFilter:
    '''
    Kalman filter class.
    Given the system dynamics matrices Ft, B; the measurement matrix H; and the noise matrices W, Mt
    '''
    def __init__(self):
        self.x_est = x0
        self.P_est = P0
        self.t = 0

    def Iterate_once(self, measurement, command):
        '''
        Estimate the state and the covariance noise
        calculate the Kalman gain using the new measurement
        Update the state and covariance estimations
        '''
        # time update
        self.t = self.t + dt
        xtt = np.matmul(Ft, self.x_est) # apriori x

        Mt = R1 + R2 / ((tf - self.t)**2)
        H = np.array([[1/(V*(tf - self.t)), 0, 0]])# measurement matrix
        self.P_est = self.calculate_p(Mt, H) # Pt = Ft*P + Pt*Ft - Pt*Htt*Mt^-1*Ht*Pt + W

        # measurement update

        K = np.matmul(self.P_est, np.transpose(H)) / Mt # K = Pt * Ht * Mt^-1

        # x_hat = (F*x_hat + Bt*command)dt + K*(dz - Ht*xtt*dt)
        self.x_est = (np.matmul(Ft, xtt) + command*np.transpose(B))*dt + np.matmul(K, measurement - np.matmul(H, xtt))

        return K, self.x_est, self.P_est

    def calculate_p(self, Mt, H):
        tmp1 = 2 * np.matmul(Ft, self.P_est)  # Ft * Pt + Pt * Ft
        tmp2 = np.matmul(self.P_est, np.transpose(H)) / Mt # Pt * Htt * Mt^-1
        tmp3 = -np.matmul(np.matmul(tmp2,H), self.P_est) + W_tilde # - Pt * Htt * Mt^-1 * Ht * Pt + W_tilde

        delta_cov = tmp1 + tmp3 #Ft * Pt + Pt * Ft - Pt * Htt * Mt^-1 * Ht * Pt + W_tilde
        return self.P_est + dt*delta_cov # Ricatti equation for Pt


    def plotCovariances(self, P_vec, ax):
        ells = []
        strethFactor = 4
        max_a=max(P_vec, key=lambda x: x[0][0])[0][0] # covariance with maximal a member
        max_c=max(P_vec, key=lambda x: x[1][1])[1][1] # covariance with maximal c member

        for i, cov in enumerate(P_vec):
            ellipse = self.covarianceToEllipse(cov, strethFactor*i, max_c)
            ells.append(ellipse)

            ax.add_artist(ellipse)
            ellipse.set_clip_box(ax.bbox)
            ellipse.set_alpha(1)
            ellipse.set_facecolor((0.6, 0.6, 0.4))

        ax.set_title('Covariances as a function of time')
        ax.set_xlabel(f"Spaced timesteps (stretched by a factor of {strethFactor})")
        ax.set_ylabel("Covariance")
        ax.set_xlim((-np.ceil(max_a), strethFactor*(len(P_vec)-1)+max_a))
        ax.set_ylim((0, max_c))

        plt.subplots_adjust(hspace=0.5)
        plt.show()

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

def estimate(filter, show=True):
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

        filter.plotCovariances(covs, ax3)

    gains = []
    states = [x0]
    covs = [P0]
    control_cmd = 0

    for i, meas in enumerate(measurements):
        K, x, P = filter.Iterate_once(meas, control_cmd)
        gains.append(K)
        states.append(x)
        covs.append(P)

    t = np.arange(len(measurements) + 1)

    gain_vecs = []
    for i in range(len(gains[0])):
        gains_xi = np.ravel([el[i] for el in gains])
        gain_vecs.append(gains_xi)

    if show:
        plot_estimates(t, gain_vecs, covs)

    return gains

def calculate_s():
    for i, t in enumerate(t_vec):
        pass

def main():
    St_list = calculate_s() #backpropagation

    kalman_filter = ContinuousKalmanFilter()

    estimate(kalman_filter, show=True)


if __name__ == '__main__':
    main()