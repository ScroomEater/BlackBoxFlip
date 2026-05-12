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

def FindMinima(reservoir, inp_fixed, z_fixed, n_candidates=10, max_iters=2000):

    N = reservoir.N
    device = reservoir.x.device

    states = torch.randn((n_candidates, N, 1), device=device, requires_grad=True)

    optimizer = optim.LBFGS([states], lr=0.1, max_iter=20)
    
    def closure():
        optimizer.zero_grad()

        r = torch.tanh(states)

        v = (-1 * states + 
             reservoir.J_layer @ r + 
             reservoir.Jf_layer @ z_fixed + 
             reservoir.B @ inp_fixed)
        
        q = 0.5 * torch.sum(v**2)
        q.backward()
        return q

    for i in range(max_iters):
        current_q = optimizer.step(closure)
        if current_q < 1e-9: # Convergence threshold
            break
            

    with torch.no_grad():
        final_r = torch.tanh(states)
        final_v = (-1 * states + reservoir.J_layer @ final_r + 
                   reservoir.Jf_layer @ z_fixed + reservoir.B @ inp_fixed)
        speeds = torch.norm(final_v, dim=(1, 2))
        
    return states.detach(), speeds
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