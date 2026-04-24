# This file will contain the code for the flip flopping object
import numpy as np
import random


"""
Essentiall find the 0s and minima of q(x) = 1/2|F(x)|^2, F(x)=x'(t)
FORCE learning algorithm? -> One hidden layer of N neurons
    Different hyperparameters, training loop
"""


class FlipFlop():
    def __init__(self, bits=3, p=0.2):
        
        #bits: number of bits to model flipping for
        #p: probabiliity of a bit flipping for one time step
        
        random.seed(1432)
        self.bits = bits
        self.p = p

        # It would be too inefficient to create a new object for each epoch
        #init_state = [-1, 1]
        #self.state = [random.choice(init_state) for _ in range(bits)]
        

    def simNSteps(self, n_steps, batch_size):
        unsigned_in = np.random.binomial(1, self.p, [batch_size, n_steps, self.bits])
        unsigned_out = 2*np.random.binomial(1, 0.5, [batch_size, n_steps, self.bits])-1

        # Creates input impulse matrix, should be mostly 0s with occational -1s and 1s
        input = np.multiply(unsigned_in, unsigned_out)

        # Initial signal of 1 so states are set to 1
        input[0, :, 0] = 1 

        # Sustained signal


        # Initialize array of 0s for output
        # Output represents state after input signal received
        output = np.zeros_like(input)

        return {"inputs": input, "outputs": output}



class FlipFlop2():
    def __init__(self):
        """
        
        """
        random.seed(323)


    def genData(self, n_steps: int, bits:int = 3, batch_size: int = 1, p: float = 0.2):
        """
        Generates test data for n_steps
        bits: number of bits to model
        p: probability that a bit flips
        returns tuple of arrays (inp, out), each with shape (batch_size, n_steps, bits)
        """

        unsigned_inp = np.random.binomial(1, p=p, size=[batch_size, n_steps, bits])  # Array of 0s and 1s, 1s with frequency p
        unsigned_out = 2 * np.random.binomial(1, p=0.5, size=[batch_size, n_steps, bits]) - 1  # Array of -1s and 1s, 50-50

        # This represents the signal to feed to the model
        inp = np.multiply(unsigned_inp, unsigned_out)  # Array of -1, 0, 1. -1/1 with frequency p, 0 with frequency 1-p
        inp[:, 0, :] = 1

        # Expected output array, shape (n_steps, bits)
        out = np.zeros_like(inp)
        for batch in range(batch_size):
            for b in range(bits):
                flip_times = np.where(inp[batch, :, b].squeeze() != 0) # An array of times the bit recieves a signal
                
                for i in range(len(flip_times[0])):
                    t = flip_times[0][i]
                    out[batch, t:, b] = inp[batch, t, b]


        return inp, out



"""
arr = np.random.binomial(1, 0.3, size=[20, 1])
print(arr)
print(np.where(arr != 0))
"""

flipTest = FlipFlop2()
inp, out = flipTest.genData(30, 3, batch_size=3)
print(inp.shape)
for b in range(len(inp)):
    for i, o in zip(inp[b], out[b]):
        print(i, o)
    
    print('\n'*3)
