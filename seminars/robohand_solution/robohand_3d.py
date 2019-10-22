

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import math
import numpy as np

from pyglet.gl import *

VIEWPORT_W, VIEWPORT_H = 800, 800
START_HEIGHT = 300
H = 1.1 * START_HEIGHT * 1.0
W = float(VIEWPORT_W) / VIEWPORT_H * H
HAND_LENGTH = 1.0

JOINT_LENGTH = 2.0
JOINT_WIDE = 0.25
SPHERE_RAD = 0.3
SPHERE_RAD_S = 0.2

def vec(*args):
    return (GLfloat * len(args))(*args)

class TRoboHand3D(gym.Env, EzPickle):
    def __init__(self, T, GAMMA = 10, DELTA = 10, moving_goal = True):
        
        self.gamma = GAMMA
        self.delta = DELTA
        self.T = T
        self.moving_goal = moving_goal
        
        #self.goal1 = np.random.random(size=2) * 360
        #self.goal2 = np.random.random(size=2) * 360
        
        self.a1 = np.array([0.0, 0.0])
        self.a2 = np.array([0.0, 0.0])
        
        self.x_size = 8
        self.u_size = 4
        
        self.light_dir = np.array([-1, -3.0, -1])
        self.color = np.array([0.2, 0.4, 0.6])
        self.goal_color = np.array([0.90, 0.90, 0.90])
        
        self.current_color = None
        self.current_diffuse = False
        
        #self.color = np.array([1,1,1])
        
        self.window = None
    
    def _init_goals(self):
        if self.moving_goal:
            self.goal_0 = np.random.random(size=4) * 360
            self.goal_1 = np.random.random(size=4) * 360
            #self.goals = np.random.random(size=(self.T,4)) * 360
            self.goals = np.zeros((self.T,4))
            #self.goals[0] = self.goal_0
            #self.goals[1] = self.goal_1
            for t in range(self.T):
                self.goals[t] = (self.goal_1 - self.goal_0) / (self.T-1) * t + self.goal_0
        else:
            self.goal0 = np.random.random(size=4) * 360
            self.goals = np.tile(self.goal0, (T,1))
    
    def _get_step_dynamic(self, step):
        
        F = np.concatenate([np.eye(self.x_size).astype(float), np.zeros((self.x_size, self.u_size)).astype(float)], 1) +\
            np.concatenate([np.zeros((self.x_size, self.u_size)).astype(float), np.eye(self.x_size).astype(float)], 1)
        
        f = np.zeros( self.x_size )
        
        C = np.diag([ 
            1, 1, 1, 1,  #for angle^2
            self.gamma, self.gamma, self.gamma, self.gamma, #velocity penanly
            self.delta, self.delta, self.delta, self.delta
        ])
        
        c = np.concatenate([ -self.goals[step], np.zeros(8) ])
        
        return F, f, C, c
    
    def _get_current_dynamic0(self):
        
        self.F = np.concatenate([np.eye(self.x_size).astype(float), np.zeros((self.x_size, self.u_size)).astype(float)], 1) +\
            np.concatenate([np.zeros((self.x_size, self.u_size)).astype(float), np.eye(self.x_size).astype(float)], 1)
        
        self.f = np.zeros( self.x_size )
        
        self.C = np.diag([ 
            1, 1, 1, 1,  #for angle^2
            self.gamma, self.gamma, self.gamma, self.gamma, #velocity penanly
            self.delta, self.delta, self.delta, self.delta
        ])
        
        self.c = np.concatenate([ -self.goal1, -self.goal2, np.zeros(8) ])
    
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
        
        self.x = np.concatenate([ self.a1, self.a2, np.zeros(4) ])
        
        self._get_current_dynamic()
        
        return self.get_dynamic()
    
    def get_dynamic( self ):
        return self.F, self.f, self.C, self.c
    
    def step(self, u):
        
        xu = np.concatenate([self.x,u])
        
        self.x = self.F[self.t].dot(xu) + self.f[self.t]
        c = xu.T.dot(self.C[self.t]).dot(xu) + self.c[self.t].dot(xu)
        self.t += 1
        
        return self.x, c, False, {}
    
    def _draw_quad( self, pts, h ):
        
        cntr = (pts[0] + pts[2]) / 2.0
        norm = cntr - np.array([0,0,h/2])
        brig = max(np.dot( norm, -self.light_dir ) * 0.3, 0.0) + 0.7
        
        clr = (self.current_color)
        #if self.current_diffuse:
        #    clr = clr * brig
        
        glColor3f( clr[0], clr[1], clr[2] )
        glBegin(GL_QUADS)
        for p in pts:
            glNormal3f( norm[0], norm[1], norm[2] )
            glVertex3f(*list(p))
        glEnd()
    
    def _draw_joint( self, w, h ):
        
        bt = np.array([
            [ -w/2.0, -w/2.0, 0.0 ],
            [ w/2.0, -w/2.0, 0.0 ],
            [ w/2.0, w/2.0, 0.0 ],
            [ -w/2.0, w/2.0, 0.0 ]
        ])
        
        tp = bt.copy()
        bt[:,2] = h
        
        #print('-'*10)
        
        #bottom
        self._draw_quad( bt, h )
        #top
        self._draw_quad( tp, h )
        #left
        self._draw_quad( np.append( bt[[1,0],:], tp[[0,1],:], 0 ), h )
        #right
        self._draw_quad( np.append( bt[[2,1],:], tp[[1,2],:], 0 ), h )
        #front
        self._draw_quad( np.append( bt[[3,2],:], tp[[2,3],:], 0 ), h )
        #back
        self._draw_quad( np.append( bt[[0,3],:], tp[[3,0],:], 0 ), h )
        
        #print('-'*10)
    
    def _draw_sphere( self, r ):
        
        n = 10
        l = math.pi * 2.0
        prev_p = np.zeros((n+1,3))
        for i in range(n+1):
            prev_p[i,:] = np.array([0,r,0])
        
        for j in range(n+1):
            y = r * math.sin( j / n / 2 * l - math.pi/2 )
            w = r * math.cos( j / n / 2 * l - math.pi/2 )
            
            glBegin(GL_QUAD_STRIP)
            
            for i in range(n+1):
                x = w * math.cos( i / n * l)
                z = w * math.sin( i / n * l)
                if j > 0:
                    
                    clr = self.current_color
                    
                    glColor3f( clr[0], clr[1], clr[2] )
                    
                    glNormal3f( x, y, z )
                    glVertex3f( x, y, z )
                    
                    glNormal3f( prev_p[i][0], prev_p[i][1], prev_p[i][2] )
                    glVertex3f( prev_p[i][0], prev_p[i][1], prev_p[i][2] )
                    
                
                prev_p[i,:] = np.array([x,y,z])
                
            glEnd()
    
    def _draw_hand(self, a1, a2, color, diffuse):
        
        self.current_color = color
        self.current_diffuse = diffuse
        
        self._set_material()
        
        ########
        #Sphere1
        #######
        self._draw_sphere( SPHERE_RAD )
        
        #######
        #Joint1
        ######
        glRotatef(a1[0], 0.0, 1.0, 0.0)
        glRotatef(a1[1], 1.0, 0.0, 0.0)
        glTranslatef( 0, 0, SPHERE_RAD/3 )
        self._draw_joint( JOINT_WIDE, JOINT_LENGTH)
        
        #########
        #Sphere2
        ########
        glTranslatef( 0, 0, JOINT_LENGTH + SPHERE_RAD/3 )
        self._draw_sphere( SPHERE_RAD )
        
        #######
        #Joint1
        #######
        glTranslatef( 0, 0, SPHERE_RAD/3 )
        glRotatef(a2[0], 0.0, 1.0, 0.0)
        glRotatef(a2[1], 1.0, 0.0, 0.0)
        self._draw_joint( JOINT_WIDE, JOINT_LENGTH )
        
        #########
        #Sphere3
        ########
        glTranslatef( 0, 0, JOINT_LENGTH + SPHERE_RAD/3 )
        self._draw_sphere( SPHERE_RAD_S )
    
    def _draw_axis(self):
        
        glTranslatef( -2, -2, -2 )
        
        glLineWidth(3)
        
        glBegin(GL_LINES)
        glColor3f(1,0,0)
        glVertex3f(0,0,0)
        glVertex3f(10,0,0)
        glEnd()
        
        glBegin(GL_LINES)
        glColor3f(0, 0, 1)
        glVertex3f(0,0,0)
        glVertex3f(0,0,10)
        glEnd()
        
        glBegin(GL_LINES)
        glColor3f(0,1,0)
        glVertex3f(0,0,0)
        glVertex3f(0,10,0)
        glEnd()
    
    def _set_lights(self):
        
        
        
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        clr = self.current_color
        
        glLightfv(GL_LIGHT0, GL_POSITION, vec(5.0, 2.0, 4.0, 0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat * 3)(4.0, 4.0, 4.0))
        glLightfv(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, (GLfloat * 1) (.35))
        #glLight( GL_LIGHT0, GL_POSITION, )
        
        
    def _set_material(self):
        
        clr = (self.current_color)
        
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, 
                     vec(clr[0], clr[1], clr[2], 1))
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(0.3, 0.3, 0.3, 1))
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 1)
        
    def render(self, mode='rgb_array', close=False):
        
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            
            return
        
        self.a1 = self.x[0:2]
        self.a2 = self.x[2:4]
        
        if self.window is None:
            self.window = pyglet.window.Window(width=VIEWPORT_W, height=VIEWPORT_H)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_DEPTH_TEST)
            glEnable( GL_LINE_SMOOTH )
            glEnable( GL_POLYGON_SMOOTH )
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            #glEnable(GL_DEPTH_TEST)
            #glEnable(GL_CULL_FACE)
        
        glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        
        glLoadIdentity()
        gluPerspective(45, 1.0, 0.1, 1000)
        gluLookAt(8,2,8, 0,0,0, 0,1,0)
        
        glPushMatrix()
        self._draw_axis()
        glPopMatrix()
        
        self._set_lights()
        #glPushMatrix()
        #self.current_color = np.array([1.0, 1.0, 0.0])
        #glTranslatef( 5.0, 2.0, 3.0 )
        #self._draw_sphere(0.2)
        #glPopMatrix()
        
        glPushMatrix()
        self._draw_hand(self.a1, self.a2, self.color, True)
        glPopMatrix()
        
        glPushMatrix()
        self._draw_hand(self.goals[self.t-1][:2], self.goals[self.t-1][2:], self.goal_color, False)
        glPopMatrix()
        
        if mode == 'rgb_array':
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.data, dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
            
            return arr
    
    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None
            
        return