import numpy as np
import matplotlib.pyplot as plt
#import sklearn
#import rnn
#import flipflop

batch_size = 10
n_steps = 10000
bits = 3
n_layers = 1
d_model = 1000

#model = rnn.RNN(d_in=bits, d_out=bits, n_layers=n_layers, d_model=d_model)
#dataGen = flipflop.FlipFlop2()

data = np.array([137, 134, 205, 19, 25, 26, 831, 108, 42.6])
print(data.mean(), data.std())