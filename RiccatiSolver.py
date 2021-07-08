import numpy as np
from scipy.integrate import odeint
import scipy as sp


class RiccatiSolver:
    """
        Solve the matrix Riccati differential equation - MRDE,
        according to following paper: https://pdf.sciencedirectassets.com/271552/1-s2.0-S0895717700X00659/1-s2.0-S0895717798000351/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEC0aCXVzLWVhc3QtMSJGMEQCIFqke1q2NB1%2Fxy1ggCGuG%2FJituFFR9SmKFka95TIt6FDAiAM9Vpme2q9SpUXz5kYZqDJCXgdaHd0GcqAv7TLP%2BHvKCr6AwgWEAQaDDA1OTAwMzU0Njg2NSIMSPYeELLoJm6DSqAtKtcDSQfxkMa9zYdMIw2W5ewARQg5OytvKC%2B8v4sU2Kb8wxjTDMle4KsOCSs0dxGHg4DpRQG3%2B6GUlJT9SOM5GqBmf7NpPtqMkYdNBu%2FcQp1ze4aiHHXj8xVUtyx%2BRwlV%2B0E2Mo8RB4Y6MAXrBqI1O3nv54slAcqC%2BdY0S%2BNRKovFJXUOVtTgiuebxErywfqq7mFCtTgXpnjvz3b5%2Ble%2FeCY%2BoYB2BriYK41lj5uSweuCEGzQfRCzEkOzIuvipGPCDTYA9jzZ%2FEsyTypFXOxSA3NCDfs0u9teHdxc5EOdov%2BUxkZb5nXBHo29XUfid809ULRMubr0vxI6VQMi5A0Uf6pCIhd1fXM7JL3wFx7fz7ehZW103Z%2Fr73KauSQSIfhcgjcOmt4EIU%2BZ5wfxuykAnZB467iqE1%2B7bZ1Hz%2FA2m%2Bei0VmMDQVgoagooRg13XV7VJmrOxkArV4la1ZFn3AyULX4PTDojVaOxtn17204R5jT8xi4KCgBIgm4E6fkiy5FRQ%2Fa%2BYTwstpqZ8dEtyEHEjtpdA81KD1FS9L9m3mKuBR4czc5iJdlzj9%2BeBTWymzRWmY9KbkQuxjx6M4TWacskGbKRsPxHEOiVe2Vd6Gm93xiDrmxoJ8kvloPMMXWhocGOqYBGLihJDacovMX1FVNwhIdWcbtpmqgkGwJ3KZ8QYhK8T9YClZgccTiEk2xRGLglBwbzWGbnDp%2FDD4gXRCXVfiXz4NODpcp1Z%2B0I68GXPbQL6tbTzh%2FdKwFps5Y%2Fmzj3li%2FxI9w4I9NUX37BJgEpagGMrQ9cDbYIrfkVIvA660k0E8tLrkRlzUs6EJmB4MRN0NkFuBor%2FlwIr5GxRBu1YoJfKFBoPb11g%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210704T124202Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY7JEGQOEV%2F20210704%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c636a5ec128fe3de4da7fe62dd152307c7916bf0ab8c2870a3fc6bbf3af6d92c&hash=3a4e189fd1c8b99c66eff9c4f18a68df95fc09e2c72982f1b4d684264c085e4f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0895717798000351&tid=spdf-cfc8705c-ab43-4b80-b7c1-31fbcec308aa&sid=8aeebf152463814e43993fc65526dd2b1facgxrqb&type=client
        d(m_S)/dt = -m_S*m_F - m_S.T*m_S + m_S*(m_B*m_R^-1*m_B.T)*m_S - m_Q
    """
    def __init__(self, m_Sf, m_F, v_B, m_Q, m_R, t):
        self.m_Sf = m_Sf
        self.m_F = m_F
        self.v_B = v_B
        self.m_Q = m_Q
        self.m_R = m_R
        self.t = t

        self.m_D1 = self.calc_m_D1()
        self.m_Ss = sp.linalg.solve_continuous_are(self.m_F, self.v_B, self.m_Q, self.m_R)
        self.m_Fc = self.calc_m_Fc()
        self.l_m_S = self.solve_matrix_Riccati_diff_eq()

    def solve_matrix_Riccati_diff_eq(self):
        m_F1 = np.kron(self.m_Fc, np.eye(self.m_Fc.shape[0])) + np.kron(np.eye(self.m_Fc.shape[0]), self.m_Fc)
        m_P0 = np.linalg.inv(self.m_Sf - self.m_Ss)
        v_P = odeint(RiccatiSolver.dP_dt, m_P0.flatten(), self.t, args=(m_F1, self.m_D1.flatten()))
        l_m_s = []
        for row in v_P:
            m_P = row.reshape(self.m_Sf.shape)
            m_S = np.linalg.inv(m_P) + self.m_Ss
            l_m_s.append(m_S)
        return l_m_s

    def calc_m_D1(self):
        if np.isscalar(self.m_R):
            return np.dot(self.v_B, self.v_B.T) / self.m_R
        return np.dot(np.dot(self.v_B.T, np.linalg.inv(self.m_R), self.v_B))

    def calc_m_Fc(self):
        if np.isscalar(self.m_D1):
            return self.m_F - self.m_D1 * self.m_Ss
        return self.m_F - np.dot(self.m_D1, self.m_Ss)

    @staticmethod
    def dP_dt(v_P, t, m_F1, v_D1):
        dPdt = - m_F1@v_P + v_D1
        return dPdt


if __name__ == '__main__':
    m_F = np.array([[0, 1], [-1, 2]])
    v_B = np.array([[0, 1]]).T
    m_Sf = np.array([[5, 0], [0, 0]])
    m_Q = np.array([[1, 0], [0, 2]])
    m_R = 1
    t = np.linspace(0, 10, 101)
    a = RiccatiSolver(m_Sf, m_F, v_B, m_Q, m_R, t)
    print(a.l_m_S)
