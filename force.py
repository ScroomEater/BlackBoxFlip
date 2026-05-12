"""
Trying to implement FORCE from Sussillo and Abbott (2009)
Discrete, not continuous-time, so it lacks some hyperparameters
"""

import numpy as np
import torch
from rnn import RNN

class ForceLearner():
    def __init__(self, model, alpha=1.0):
        """
        model: the RNN to train on
        alpha: the learning rate initializer
        """
        self.model = model
        self.P = torch.eye(len(model)) * (1/alpha)

    def step(self, z_pre, out):
        error = z_pre - out

        self.P = (self.P - (self.P @ self.model.r @ self.model.r.T @ self.P)/
                (1 + self.model.r.T @ self.P @ self.model.r))

        self.model.w = self.model.w - self.P @ self.model.r @ error.T

        return self.model.w.T @ self.model.r


class Reservoir():
    def __init__(self, d_in, d_out, N, g=1.2):
        self.N = N
        self.J_layer = torch.randn((N, N)) * g / np.sqrt(N)
        self.Jf_layer = (torch.randn((N, d_out)) * 2.0) - 1.0
        self.B = (torch.randn((N, d_in)) * 2.0) - 1.0
        self.w = torch.randn((N, d_out))

        self.x = torch.zeros((N, 1))
        self.r_activ = torch.nn.Tanh()
        self.r = self.r_activ(self.x)

    def forward(self):  
        self.r = self.r_activ(self.x)
        z_pre = self.w.T @ self.r
        return z_pre
    
    def update(self, inp, z_post):
        dx = -1 * self.x + self.J_layer @ self.r + self.Jf_layer @ z_post + self.B @ inp
        self.x = self.x + dx

    def change_state(self, x):
        self.x = x
        self.r = self.r_activ(x)

    def __len__(self):
        return self.N


"""
testing 
example training loop:
model = Reservoir(i, o, N, g)
learner = ForceLearner(model, a)



for i in timesteps:
    z_pre = model.forward()

    learner.step(z_pre, outputs[i])

    model.update(inputs[i])
"""