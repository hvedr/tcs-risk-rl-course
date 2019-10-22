import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import math
import numpy as np

class LQRController:
    def __init__(self, T, s_size, u_size):
        self.T = T
        self.s_size = s_size
        self.u_size = u_size
        self.a_seq = []
        self.s_seq = []
    
    def Q(u, x, Q, q):
        ux = np.concatenate(x,u)
        return 1/2*np.dot(np.dot(Q, ux), ux.T) + np.dot(ux)*q
    
    def find_action(self, F, f, C, c):
        self.A = F[:,:-2]
        self.B = F[:,-2:]
        self.Q = C[:-2, :-2]
        self.R = C[-2:, -2:]
        
        K, Ig1, Ig2 = dlqr(self.A, self.B, self.Q, self.R)
        
        return K
    
    def find_trajectory(self, F, f, C, c, s0):
        A = F[:, :self.s_size]
        B = F[:, self.s_size:]
        Q = C[:self.s_size, :self.s_size]
        R = C[self.s_size:, self.s_size:]
        Ps = np.zeros( (self.T+1, self.s_size, self.s_size) )
        Fs = np.zeros( (self.T, self.u_size, self.s_size) )
        
        print('A', A.shape, 'B', B.shape, 'Q', Q.shape, 'R', R.shape)
        
        Ps[self.T,:,:] = Q
        
        for t_ in range(self.T):
            t = self.T - 1 - t_
            Pk = Ps[t+1,:,:]
            II = R+B.T.dot(Pk).dot(B)
            print(II)
            II = np.linalg.inv( II.astype(float) + np.random.random(II.shape)*0.0001 )
            
            Ps[t,:,:] = A.T.dot(Pk) - (A.T.dot(Pk).dot(B)).\
                dot(II).dot(B.T.dot(Pk).dot(A))+Q
            
            II = R+B.T.dot(Pk).dot(B)
            II = np.linalg.inv( II + np.random.random(II.shape)*0.0001 )
            
            Fs[t,:,:] = II.dot(B.T.dot(Pk).dot(A))
        
        x_t = s0
        us = np.zeros((self.T, self.u_size))
        for t in range(self.T):
            us[t, :] = -np.dot(Fs[t,:,:], x_t)
            x_t = np.dot( A, x_t ) + B.dot(us[t,:])
        
        return us
    
    def find_trajectory1(self, F, f, C, c, s0):
        
        self.F = F
        self.f = f
        self.C = C
        self.c = c
        
        T = self.T
        Qs = np.zeros((T, self.C[0].shape[0], self.C[0].shape[1]))
        qs = np.zeros((T, self.s_size+self.u_size))
        Vs = np.zeros((T+1, self.s_size, self.s_size))
        vs = np.zeros((T+1, self.s_size))
        us = np.zeros((T, self.u_size))
        ks = np.zeros((T, self.u_size))
        Ks = np.zeros((T, self.u_size, self.s_size))
        
        for t_ in range(self.T):
            
            t = self.T - 1 - t_
            
            self.F = F[t]
            self.f = f[t]
            self.C = C[t]
            self.c = c[t]
            
            Qs[t,:,:] = self.C + self.F.T.dot(Vs[t+1].dot(self.F))
            qs[t,:] = self.c + self.F.T.dot(Vs[t+1]).dot(self.f) + np.dot(self.F.T, vs[t+1,:])
            
            Qxx = Qs[t,:self.s_size, :self.s_size]
            Qxu = Qs[t,:self.s_size, self.s_size:]
            Qux = Qs[t,self.s_size:, :self.s_size]
            Quu = Qs[t,self.s_size:, self.s_size:]
            qx = qs[t, :self.s_size]
            qu = qs[t, self.s_size:]
            
            #print(
            #    'Qxx', Qxx.shape,
            #    'Qxu', Qxu.shape,
            #    'Qux', Qux.shape,
            #    'Quu', Quu.shape,
            #    'qx', qx.shape,
            #    'qu', qu.shape
            #)
            
            K_t = -np.dot(
                np.linalg.inv( Quu + np.random.random(Quu.shape)*0.00000 ),
                Qux
            )
            k_t = -np.dot(
                np.linalg.inv( Quu + np.random.random(Quu.shape)*0.00000 ), 
                qu
            )
            
            Vs[t,:,:] = Qxx + np.dot(Qxu, K_t) + np.dot(K_t.T, Qux) +\
                K_t.T.dot(Quu).dot(K_t)
            
            vs[t,:] = qx + np.dot(Qxu, k_t) + np.dot(K_t.T, qu) +\
                K_t.T.dot(Quu).dot(k_t)
            
            Ks[t,:,:] = K_t
            ks[t,:] = k_t
        
        self.Qs = Qs
        self.Vs = Vs
        self.qs = qs
        self.vs = vs
        self.Ks = Ks
        self.ks = ks
        
        #print(s0.shape, us[0].shapem, Ks[0].shape, ks)
        x_t = s0
        for t in range(self.T):
            us[t, :] = np.dot(Ks[t,:,:], x_t) + ks[t,:]
            #print('k', t, Ks[t,:,:])
            x_t = np.dot( F[t], np.concatenate([x_t, us[t,:]]) ) + f[t]
            #print('u', t, us[t, :])
            
        return us
        
    def backward_pass(self):
        pass
    
    def forward_pass(self):
        pass
    

 