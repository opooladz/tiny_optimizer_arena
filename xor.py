#@title RNN network with the classic delayed XOR problem. Lets see how PSGD Kron does

import matplotlib.pyplot as plt
import numpy as np
import torch
from kron import Kron

device = torch.device('cuda:0')
batch_size, seq_len = 128, 57          # increasing sequence_length or decreasing dimension_hidden_layer will make learning harder;
dim_in, dim_hidden, dim_out = 2, 30, 1  # current setting can solve seq len 80 ~ 90 reliably without the help of momentum
def generate_train_data():
    x = np.zeros([batch_size, seq_len, dim_in], dtype=np.float32)
    y = np.zeros([batch_size, dim_out], dtype=np.float32)
    for i in range(batch_size):
        x[i, :, 0] = np.random.choice([-1.0, 1.0], seq_len)

        i1 = int(np.floor(np.random.rand() * 0.1 * seq_len))
        i2 = int(np.floor(np.random.rand() * 0.4 * seq_len + 0.1 * seq_len))
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        if x[i, i1, 0] == x[i, i2, 0]:  # XOR
            y[i] = -1.0  # lable 0
        else:
            y[i] = 1.0  # lable 1

    # tranpose x to format (sequence_length, batch_size, dimension_of_input)
    return [torch.tensor(np.transpose(x, [1, 0, 2])).to(device),
            torch.tensor(y).to(device)]

# generate a random orthogonal matrix for recurrent matrix initialization
def get_rand_orth(dim):
    temp = np.random.normal(size=[dim, dim])
    q, _ = np.linalg.qr(temp)
    return torch.tensor(q, dtype=torch.float32).to(device)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.W1x = torch.nn.Parameter(0.1 * torch.randn(dim_in, dim_hidden))
        self.W1h = torch.nn.Parameter(get_rand_orth(dim_hidden))
        self.b1 = torch.nn.Parameter(torch.zeros(dim_hidden))
        self.W2 = torch.nn.Parameter(0.1 * torch.randn(dim_hidden, dim_out))
        self.b2 = torch.nn.Parameter(torch.zeros([]))

    def forward(self, xs):
        h = torch.zeros(batch_size, dim_hidden, device=device)
        for x in torch.unbind(xs):
            h = torch.tanh(x @ self.W1x + h @ self.W1h + self.b1)
        return h @ self.W2 + self.b2
model = Model()
# @torch.compile(backend='cudagraphs')

# model = torch.compile(model, )
model = Model().to(device)

# initialize the PSGD optimizer with the vanilla Kron preconditioner
opt = Kron(model.parameters(), lr=0.00008,b1=0.9,max_skew_triangular=torch.inf,max_size_triangular=torch.inf,preconditioner_update_probability=1)

def train_loss(xy_pair):  # logistic loss
    return -torch.mean(torch.log(torch.sigmoid(xy_pair[1] * model(xy_pair[0]))))

Losses = []
for num_iter in range(20000):
    train_data = generate_train_data()

    loss = train_loss(train_data) # return a loss
    # Backward and optimize
    opt.zero_grad()
    loss.backward()
    opt.step()
    Losses.append(loss.item())
    print('Iteration: {}; loss: {}'.format(num_iter, Losses[-1]))

    if Losses[-1] < 0.1:
        print('Deemed to be successful and ends training')
        break
plt.plot(Losses)
