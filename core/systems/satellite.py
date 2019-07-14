from core.dynamics import RoboticDynamics, SystemDynamics
from .util import attitude_utils

from numpy import concatenate, dot, identity, reshape, tensordot, zeros
from numpy.linalg import inv, solve

class Satellite(SystemDynamics, RoboticDynamics):
    def __init__(self, J, rot_order):
        SystemDynamics.__init__(self, 6, 3)
        RoboticDynamics.__init__(self, identity(3))
        self.J = J
        self.J_inv = inv(J)
        self.dcm, self.graddcm, self.hessdcm = attitude_utils.dcm_from_euler(rot_order)
        self.T, self.grad_T = attitude_utils.euler_to_ang(rot_order)

    def drift(self, x, t):
        xi, xi_dot = reshape(x, (2, 3))
        J = self.J
        T = self.T(xi)
        omega = dot(T, xi_dot)
        return concatenate([xi_dot, -solve(T, solve(J, dot(attitude_utils.ss_cross(omega), dot(J, omega))) - dot(tensordot(self.grad_T(xi), xi_dot, (-1, 0)), xi_dot))])

    def act(self, x, t):
        xi, _ = reshape(x, (2, 3))
        return concatenate([zeros((3, 3)), solve(self.T(xi), self.J_inv)])
