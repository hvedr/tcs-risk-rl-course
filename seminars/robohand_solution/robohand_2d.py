import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import math
import numpy as np

VIEWPORT_W, VIEWPORT_H = 800, 800
START_HEIGHT = 300
H = 1.1 * START_HEIGHT * 1.0
W = float(VIEWPORT_W) / VIEWPORT_H * H
HAND_LENGTH = 1.0

class TRoboHand2D(gym.Env, EzPickle):
    def __init__(self, T, GAMMA = 10, DELTA = 10, moving_goal = True):
        
        self.gamma = GAMMA
        self.delta = DELTA
        self.T = T
        self.moving_goal = moving_goal
        
        self.a1 = np.array([0.0])
        self.a2 = np.array([0.0])
        
        self.x_size = 4
        self.u_size = 2
        
        self.color = np.array([0.2, 0.4, 0.6])
        self.goal_color = np.array([0.90, 0.90, 0.90])
        
        self.current_color = None
        self.current_diffuse = False
        
        self.viewer = None
    
    def _init_goals(self):
        mlt = 3.14 * 2.0
        if self.moving_goal:
            self.goal_0 = np.random.random(size=2) * mlt
            self.goal_1 = np.random.random(size=2) * mlt
            self.goals = np.zeros((self.T,2))
            for t in range(self.T):
                self.goals[t] = (self.goal_1 - self.goal_0) / (self.T-1) * t + self.goal_0
        else:
            self.goal0 = np.random.random(size=4) * mlt
            self.goals = np.tile(self.goal0, (T,1))
    
    def _get_step_dynamic(self, step):
        
        F = np.concatenate([np.eye(self.x_size).astype(float), np.zeros((self.x_size, self.u_size)).astype(float)], 1) +\
            np.concatenate([np.zeros((self.x_size, self.u_size)).astype(float), np.eye(self.x_size).astype(float)], 1)
        
        f = np.zeros( self.x_size )
        
        C = np.diag([ 
            1, 1,
            self.gamma, self.gamma, #velocity penanly
            self.delta, self.delta,
        ])
        
        c = np.concatenate([ -self.goals[step], np.zeros(4) ])
        
        return F, f, C, c
    
    def _get_current_dynamic(self):
        
        self.F = []
        self.f = []
        self.C = []
        self.c = []
        
        for s in range(self.T):
            F, f, C, c = self._get_step_dynamic(s)
            self.F += [F]
            self.f += [f]
            self.C += [C]
            self.c += [c]
    
    def reset(self):
        self._init_goals()
        
        self.t = 0
        
        self.x = np.concatenate([ self.a1, self.a2, np.zeros(2) ])
        
        self._get_current_dynamic()
        
        return self.get_dynamic()
    
    def get_dynamic( self ):
        return self.F, self.f, self.C, self.c
    
    def step(self, u):
        xu = np.concatenate([self.x, u])
        
        self.x = self.F[self.t].dot(xu) + self.f[self.t]
        c = xu.T.dot(self.C[self.t]).dot(xu) + self.c[self.t].dot(xu)
        self.t += 1
        
        return self.x, c, False, {}
        
    def render(self, mode='rgb_array', close=False):
        
        self.a1 = self.x[0]
        self.a2 = self.x[1]
        self.goal1 = self.goals[self.t-1][0]
        self.goal2 = self.goals[self.t-1][1]
        
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            
            return
        
        def make_joint(clrs):
            joint1 = rendering.make_capsule(HAND_LENGTH, 0.2)
            joint1.set_color(*clrs)
            joint1_trans = rendering.Transform()
            joint1.add_attr(joint1_trans)
            self.viewer.add_geom(joint1)
            return joint1, joint1_trans
        
        if self.viewer is None:
        
            from gym.envs.classic_control import rendering
            
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds( -2, 2, -2, 2 )
            
            joint1s, self.joint1s_trans = make_joint((0.95, 0.95, 0.95))
            joint2s, self.joint2s_trans = make_joint((0.95, 0.95, 0.95))
            
            joint1, self.joint1_trans = make_joint((0.7, 0.3, 0.3))
            joint2, self.joint2_trans = make_joint((0.7, 0.3, 0.3))
            
        
        self.joint1_trans.set_rotation( self.a1 )
        self.joint2_trans.set_translation( math.cos(self.a1)*HAND_LENGTH, math.sin(self.a1)*HAND_LENGTH )
        self.joint2_trans.set_rotation( self.a1 + self.a2 )
        
        self.joint1s_trans.set_rotation( self.goal1 )
        self.joint2s_trans.set_translation( math.cos(self.goal1)*HAND_LENGTH, math.sin(self.goal1)*HAND_LENGTH )
        self.joint2s_trans.set_rotation( self.goal1 + self.goal2 )
        
        return self.viewer.render(return_rgb_array = (mode == 'rgb_array') )
    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None