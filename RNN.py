# This file is for a basic recurrent neural network
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class RNN(nn.Module):
    """
    d_in: dimension of input data
    d_out: output dimension
    n_layers: the number of hidden layers
    d_model: either an int representing the dimension of the hidden layers,
        or a list of hidden layers
    """
    def __init__(self, d_in, d_out, n_layers: int = 1, d_model: int | list =64):
        """
        d_model supports list for layers of different dimension or int for constant dim
        """
        super().__init__()
        
        # TODO consider adding layer_norm

        if isinstance(d_model, list) or isinstance(d_model, np.ndarray):
            if n_layers != len(d_model):
                raise ValueError("The number of layers should be equal to the size of d_model")
            
            last_dim = d_in
            W_ih = []
            W_hh = []
            self.hidden_size = []
            for dim in d_model:
                W_ih.append(nn.Linear(last_dim, dim))
                W_hh.append(nn.Linear(dim, dim))
                self.hidden_size.append(dim)
                last_dim = dim
            
            self.W_ih = nn.ModuleList(W_ih)
            self.W_hh = nn.ModuleList(W_hh)
            

            self.output_layer = nn.Linear(last_dim, d_out)

            
        else:
            # W_ih: list of connections from previous layers' hidden states
            # W_hh: list of connections from this layer's hidden state from past timestep

            self.W_ih = nn.ModuleList([nn.Linear(d_in, d_model)]
                                        + [nn.Linear(d_model, d_model) for _ in range(1, n_layers)])

            self.W_hh = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
            self.hidden_size = d_model
            self.output_layer = nn.Linear(d_model, d_out)
            

        self.n_layers = n_layers

        # self.activation = nn.ReLU()
        self.activation = nn.Tanh() # apparently better for RNN, prob bc vanishing gradient



    def forward(self, data, h_0=None):
        batch_size, n_obs, features = data.shape

        if h_0 is None:
            if type(self.hidden_size) == int:
                h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=data.device)
            else:
                h_0 = []
                for size in self.hidden_size:
                    h_0.append(torch.zeros(batch_size, size))

        # Iterate over the first dimension (n_layers)
        prev_state = list(h_0) if type(self.hidden_size) == int else h_0 # Don't quite understand this?
        preds = []

        for t in range(n_obs):
            prev_out = data[:, t, :]
            for i in range(self.n_layers):
                output = self.activation(self.W_ih[i](prev_out) + self.W_hh[i](prev_state[i]))
                prev_out = output
                prev_state[i] = output

            preds.append(self.output_layer(prev_out))

        return torch.stack(preds, dim=1) # Change to torch tensor, apparently stack() is faster than tensor()




# Review this again
def train(
    model: RNN,
    data: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    seq_len: int = 32,
) -> list[float]:
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
    if data.dim() == 1:
        data = data.unsqueeze(-1)

    if targets is None:
        # Default: next-step prediction
        inputs = data[:-1]
        targets = data[1:]
    else:
        if isinstance(targets, np.ndarray):
            targets = torch.tensor(targets, dtype=torch.float32)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        inputs = data

    # Split into non-overlapping windows: (n_windows, seq_len, features/d_out)
    n_obs = inputs.shape[0]
    n_windows = n_obs // seq_len
    inputs  = inputs[:n_windows * seq_len].reshape(n_windows, seq_len, -1)
    targets = targets[:n_windows * seq_len].reshape(n_windows, seq_len, -1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(n_windows)

        for i in range(0, n_windows, batch_size):
            idx = perm[i : i + batch_size]
            x, y = inputs[idx], targets[idx]

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / max(1, n_windows // batch_size))

    return losses


def trainFORCE(model, data, targets, initial, chaos, int_ts):
    pass

# Random normal data:
data = np.random.rand(100000, 1)
model = RNN(1, 1, n_layers=1, d_model=1000)

losses = train(model=model, data=data)

plt.plot(losses)
plt.show()

