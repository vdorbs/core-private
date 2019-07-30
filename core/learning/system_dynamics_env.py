from gym import Env
from gym.envs.classic_control.rendering import Viewer
from gym.spaces import Box
from numpy import Inf

class SystemDynamicsEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, system_dynamics, init_state, times, animator=None, atol=1e-6, rtol=1e-6):
        self.system_dynamics = system_dynamics
        self.times = times
        self.animator = animator
        self.atol = atol
        self.rtol = rtol

        self.observation_space = Box(-Inf, Inf, (system_dynamics.n,))
        self.action_space = Box(-Inf, Inf, (system_dynamics.m,))

        self.state = None
        self.time_step = None

        self.viewer = None

    def step(self, action):
        assert(self.action_space.contains(action))

        state = self.state
        initial_time = self.times[self.time_step]
        final_time = self.times[self.time_step + 1]
        self.state = self.system_dynamics.step(state, action, initial_time, final_time)

        assert self.observation_space.contains(self.state)

        self.time_step += 1
        return self.state, 0, False, {}

    def reset(self):
        self.state = init_state
        self.time_step = 0
        return self.state

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer(500, 500)
            self.animator.set_viewer(self.viewer)
        self.animator.render(self.state)
        self.viewer.render()

    def close(self):
        if self.viewer:
            self.viewer.close()
        self.viewer = None
