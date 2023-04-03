### Loudspeaker model class

import numpy as np
import math
import scipy.io as sio
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

class Transducer:
    
    def __init__(self, Re = 3.54, Le = 1.45e-4, Bl = 2.49, Mms = 2.8e-3, Cms = 4.65e-4, Rms = 0.65, fs = 48000):
        
        self.Re = Re # Ohm
        self.Le = Le # H
        self.Bl = Bl # N/A
        self.Mms = Mms # kg
        self.Cms = Cms # m/N
        self.Kms = 1/Cms # N/m
        self.Rms = Rms #kg/s
        
        self.f_res = 1/math.pi*math.sqrt(1./(self.Mms*self.Cms))
        
        ## Digital parameters
        
        self.fs = fs
        self.Ts = 1/self.fs
        
        ## State-space matrices
        self.Ad = self.get_state_matrix()
        self.Bd = self.get_input_matrix()
        self.Cd = self.get_output_matrix()
        self.Dd = self.get_feedforward_matrix()
        
        ## Internal state vector
        
        self.state = [0,0,0]
        
        
    ### Digital state space model
    def get_state_matrix(self):
        
        return np.array([[1-self.Ts*self.Re/self.Le,     -self.Ts*self.Bl/self.Le,         0],
                         [self.Ts*self.Bl/self.Mms,     1-self.Ts*self.Rms/self.Mms,   -self.Ts*self.Kms/self.Mms],
                         [0,                    self.Ts,             1]])
    
    def get_input_matrix(self):
        
        return np.array([self.Ts/self.Le,  0,  0])
    
    def get_output_matrix(self):
        
        return np.array([[1,   0,   0],
               			 [0,   1,   0],
               			 [0,   0,   1]])
    
    def get_feedforward_matrix(self):

        return np.array([0,0,0])
    
    ### Getting speaker responces
    
    def get_response(self,signal,responce):
        
        resp = np.zeros([len(signal),3]) #initial state vector
        
        for i in range(len(signal)-1):
            resp[i+1] = np.matmul(self.Ad,resp[i]) + self.Bd*signal[i]
            
        self.state = resp[-1,:]
            
        if responce == 'current':
            return resp[:,0]
        
        elif responce == 'velocity':
            return resp[:,1]
        
        elif responce == 'displacement':
            return resp[:,2]
        
        elif responce == 'training_data':
            resp[:,1] = resp[:,0].copy()
            resp[:,2] = resp[:,2].copy()
            resp[:,0] = signal.copy()
            return resp
        
        else:
            print('Choose available loudspeaker response')
        
    def reset_states(self):
        
        self.states = [0,0,0]

    
    

class TransducerNonlin:
    
    def __init__(self, Re = 3.54, Le = [0,0,1.45e-4], Bl = [0,0,2.49], Mms = 2.8e-3, Cms = [0,0,4.65e-4], Rms = 0.65, fs = 48000, stateless = True):
        
        self.Re = Re # Ohm
        self.Le_x = np.array(Le) # H
        Le_flip = np.flip(self.Le_x)
        self.Le_x_prime = np.flip(np.array([Le_flip[i] * i for i in range(1, len(Le))]))
        self.Bl_x = np.array(Bl) # N/A
        self.Mms = Mms # kg
        self.Cms_x = np.array(Cms) # m/N
        #self.Kms_x = 1/self.Cms_x # N/m
        self.Rms = Rms #kg/s
        
        self.f_res = 1/math.pi*math.sqrt(1./(self.Mms*self.Cms_x[-1]))
        
        ## Digital parameters
        
        self.fs = fs
        self.Ts = 1/self.fs
        
        ## Internal state vector
        
        self.state = [0,0,0]
        
        
    ### Digital state space model
    def get_state_matrix(self):
        
        Bl = np.polyval(self.Bl_x, self.state[2])

        # if not math.isnan(Bl):
        #     print(self.state[2])
            
        Cms = np.polyval(self.Cms_x, self.state[2])
        Le = np.polyval(self.Le_x, self.state[2])
        Le_prime = np.polyval(self.Le_x_prime, self.state[2])
        
        return np.array([[           1-self.Ts*self.Re/Le,                         -self.Ts*(self.state[0]*Le_prime + Bl)/Le,                     0],
                         [self.Ts*(Bl+self.state[0]*0.5*Le_prime)/self.Mms,               1-self.Ts*self.Rms/self.Mms,                -self.Ts/(Cms*self.Mms)],
                         [                  0,                                                     self.Ts,                                       1]])
    
    def get_input_matrix(self):
        
        Le = np.polyval(self.Le_x, self.state[2])
        
        return np.array([self.Ts/Le,  0,  0])
    
    def get_output_matrix(self):
        
        return np.array([[1,   0,   0],
               			 [0,   1,   0],
               			 [0,   0,   1]])
    
    def get_feedforward_matrix(self):

        return np.array([0,0,0])
    
    ### Getting speaker responces
    
    def get_response(self,signal, responce = None):

        resp = np.zeros([len(signal),3]) #initial state vector
        
        for i in range(len(signal)-1):
            
            Ad = self.get_state_matrix()
            Bd = self.get_input_matrix()
            Cd = self.get_output_matrix()
            Dd = self.get_feedforward_matrix()
            
            #print(signal[i])
            state_new = np.matmul(Ad,self.state) + Bd*signal[i]
            resp[i+1] = state_new
            
            self.state = state_new
            
        if responce == 'current':
            return resp[:,0]
        
        elif responce == 'velocity':
            return resp[:,1]
        
        elif responce == 'displacement':
            return resp[:,2]
        
        elif responce == 'training_data':
            resp[:,1] = resp[:,0].copy()
            resp[:,2] = resp[:,2].copy()
            resp[:,0] = signal.copy()
            return resp
        
        elif responce == None:
            
            return resp
            
        
        else:
            print('Choose available loudspeaker response')
        
        if stateless:
            
            self.reset_states()
            
            
            
    def show_Bl(self, dc_shift, bl_res, xlim = [-10,10]):

        x = np.linspace(xlim[0], xlim[1], 20)
        Bl_x = np.polyval(self.Bl_x, x)

        plt.figure()

        plt.scatter(x, Bl_x, s=100, alpha=0.5, facecolors='none', edgecolors='b', label='bl_x')

        plt.plot(dc_shift, bl_res, 'r', label='dc_shift vs. bl')

        plt.grid()
        plt.xlabel('Membrane displacement, m')
        plt.ylabel('Bl, N/A')
        plt.title('Nonlinear force factor Bl')
        plt.legend()
        plt.show()
        
        
    def show_Cms(self,xlim = [-10,10], Kms = False):
        
        x = np.linspace(xlim[0],xlim[1],20)
        Cms_x = np.polyval(self.Cms_x, x)
        
        plt.figure()
        plt.grid()
        plt.xlabel('Membrane displacement, m')
        
        if Kms: 
            plt.plot(x,1/Cms_x)
            plt.ylabel('Kms, N/m')
            plt.title('Nonlinear stiffness Kms')
        else:
            plt.plot(x,Cms_x)
            plt.ylabel('Cms, m/N')
            plt.title('Nonlinear compliance Cms')


    
    def show_Le(self,xlim = [-10,10]):
        
        x = np.linspace(xlim[0],xlim[1],20)
        Le_x = np.polyval(self.Le_x, x)
        
        plt.figure()
        plt.plot(x,Le_x)
        plt.grid()
        plt.xlabel('Membrane displacement, m')
        plt.ylabel('Le, Hn')
        plt.title('Nonlinear voice coil inductance Le')
          
        
        
    def reset_states(self):
        
        self.state = [0,0,0]
