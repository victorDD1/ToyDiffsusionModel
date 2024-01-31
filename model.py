import torch
import torch.nn as nn

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, 1, self.num_out) * out
        return out

class ConditionalMLP(nn.Module):
    def __init__(self,
                 dim_in:int,
                 dim_out:int,
                 n_steps:int,
                 hidden_layers:int=1,
                 hidden_dim:int=32,
                 activation:nn.Module=nn.LeakyReLU()) -> None:
        super(ConditionalMLP, self).__init__()

        self.in_layer = ConditionalLinear(dim_in, hidden_dim, n_steps)
        self.hidden_layers = [ConditionalLinear(hidden_dim, hidden_dim, n_steps) for _ in range(hidden_layers)]
        self.out_layer = nn.Linear(hidden_dim, dim_out)

        self.activation = activation

    def forward(self, x, t) -> torch.Tensor:
        x = self.activation(self.in_layer(x, t))
        for layer in self.hidden_layers:
            x = self.activation(layer(x, t))
        x = self.out_layer(x)
        return x