import numpy as np
from scipy.integrate import odeint
import scipy as sp


class RiccatiSolver:
    """
        Solve the matrix Riccati differential equation - MRDE,
        d(m_S)/dt = -m_S*m_F - m_S.T*m_S + m_S*(m_B*m_R^-1*m_B.T)*m_S - m_Q
    """
    def __init__(self, m_S0, m_F, v_B, m_Q, m_R, t):
        self.m_S0 = m_S0
        self.m_F = m_F
        self.v_B = v_B
        self.m_Q = m_Q
        self.m_R = m_R
        self.t = t

        self.l_m_S = self.solve_matrix_Riccati_diff_eq()

    def solve_matrix_Riccati_diff_eq(self):
        v_S = odeint(RiccatiSolver.dS_dt, self.m_S0.flatten(),
                     self.t, args=(self.m_F, self.v_B, self.m_Q, self.m_R, self.m_S0.shape))
        l_m_s = []
        for row in v_S:
            m_S = row.reshape(self.m_S0.shape)
            l_m_s.append(m_S)
        return l_m_s

    @staticmethod
    def dS_dt(v_S0, t, m_F, v_B, m_Q, m_R, shape):
        m_S = v_S0.reshape(shape)
        if np.isscalar(m_R):
            dSdt = - m_S @ m_F - m_S.T @ m_F + m_S @ v_B @ v_B.T @ m_S / m_R - m_Q
        else:
            dSdt = - m_S@m_F - m_S.T@m_F + m_S@v_B@np.linalg.inv(m_R)@v_B.T@m_S - m_Q
        return dSdt.flatten()


if __name__ == '__main__':
    m_F = np.array([[0, 1], [-1, 2]])
    v_B = np.array([[0, 1]]).T
    m_Sf = np.array([[5, 0], [0, 0]])
    m_Q = np.array([[1, 0], [0, 2]])
    m_R = 1
    t = np.linspace(0, 10, 101)
    a = RiccatiSolver(m_Sf, m_F, v_B, m_Q, m_R, t)
    print(a.l_m_S)
