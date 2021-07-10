from scipy.integrate import odeint
from consts import *


class RiccatiSolver:
    """
        Solve the matrix Riccati differential equation - MRDE,
        d(m_S)/dt = -m_S*m_F - m_S.T*m_S + m_S*(m_B*m_R^-1*m_B.T)*m_S - m_Q
    """
    @staticmethod
    def solve_Riccati_S(m_S0, m_F, v_B, m_Q, m_R, t):
        v_S = odeint(RiccatiSolver.dS_dt, m_S0.flatten(),
                     t, args=(m_F, v_B, m_Q, m_R, m_S0.shape))
        l_m_s = []
        for row in v_S:
            m_S = row.reshape(m_S0.shape)
            l_m_s.append(m_S)
        return l_m_s

    @staticmethod
    def dS_dt(v_S0, t, m_F, v_B, m_Q, m_R, shape):
        m_S = v_S0.reshape(shape)
        if np.isscalar(m_R):
            dSdt = - m_S @ m_F - m_F.T @ m_S + m_S @ v_B @ v_B.T @ m_S / m_R - m_Q
        else:
            dSdt = - m_S @ m_F - m_F.T @ m_S + m_S@v_B@np.linalg.inv(m_R)@v_B.T@m_S - m_Q
        return dSdt.flatten()

    @staticmethod
    def solve_Riccati_P(m_P0, m_F, m_W, t, tf):
        v_P = odeint(RiccatiSolver.dP_dt, m_P0.flatten(),
                     t, args=(m_F, m_W, m_P0.shape, tf))
        l_m_P = []
        for row in v_P:
            m_P = row.reshape(m_P0.shape)
            l_m_P.append(m_P)
        return l_m_P

    @staticmethod
    def dP_dt(v_P0, t, m_F, m_W, shape, tf):
        v_H = np.array([[1 / (V * (tf - t)), 0, 0]])
        M = R1 + R2 / ((tf - t) ** 2)
        m_P = v_P0.reshape(shape)
        dPdt = m_F @ m_P + m_P @ m_F.T - m_P @ v_H.T @ v_H @ m_P / M + m_W
        return dPdt.flatten()


if __name__ == '__main__':
    m_F = np.array([[0, 1], [-1, 2]])
    v_B = np.array([[0, 1]]).T
    m_Sf = np.array([[5, 0], [0, 0]])
    m_Q = np.array([[1, 0], [0, 2]])
    m_R = 1
    t = np.linspace(0, 10, 101)
    a = RiccatiSolver(m_Sf, m_F, v_B, m_Q, m_R, t)
    print(a.l_m_S)
