import pytest
import torch

import torch_optimizer as optim

# not used but worked before -- adahessian did v well.
def StochasticRosenbrock(xs):
    x1, x2 = xs
    x1 = random.uniform(0, 1) #  u can adjust the style of noise added in
    return 100.0*(x2 - x1**2)**2 + (1.0 - x1)**2

def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 1 * (y - x**2) ** 2


def quadratic(tensor):
    x, y = tensor
    a = 1.0
    b = 1.0
    return (x**2) / a + (y**2) / b
def rastrigin(tensor, lib=torch):
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization
    x, y = tensor
    A = 10
    f = (
        A * 2
        + (x**2 - A * lib.cos(x * math.pi * 2))
        + (y**2 - A * lib.cos(y * math.pi * 2))
    )
    return f

def beale(tensor):
    x, y = tensor
    f = (
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y**2) ** 2
        + (2.625 - x + x * y**3) ** 2
    )
    return f


cases = [
    (rosenbrock, (1.5, 1.5), (1, 1)),
    (quadratic, (1.5, 1.5), (0, 0)),
    (beale, (1.5, 1.5), (3, 0.5)),
    (rastrigin,(1.5, 1.5), (-2.0, 3.5))
]


def ids(v):
    n = "{} {}".format(v[0].__name__, v[1:])
    return n


def build_lookahead(*a, **kw):
    base = optim.Yogi(*a, **kw)
    return optim.Lookahead(base)
from torch.optim import *
import torch
import math
import numpy as np
optimizers = [
    (optim.A2GradUni, {"lips": 40, "beta": 0.0001}, 800),
    (optim.PID, {"lr": 0.002, "momentum": 0.8, "weight_decay": 0.0001}, 900),
    (optim.QHM, {"lr": 0.02, "momentum": 0.95, "nu": 1}, 900),
    (
        optim.NovoGrad,
        {"lr": 2.9, "betas": (0.9, 0.999), "grad_averaging": True},
        900,
    ),
    (optim.RAdam, {"lr": 0.01, "betas": (0.9, 0.95), "eps": 1e-3}, 5000),
    (optim.SGDW, {"lr": 0.002, "momentum": 0.91}, 900),
    (optim.DiffGrad, {"lr": 0.5}, 500),
    (optim.AdaMod, {"lr": 1.0}, 800),
    (optim.AdaBound, {"lr": 1.0}, 5000),
    (optim.Yogi, {"lr": 1.0}, 5000),
    (optim.AccSGD, {"lr": 0.015}, 800),
    (build_lookahead, {"lr": 1.0}, 500),
    (optim.QHAdam, {"lr": 1.0}, 500),
    (optim.AdamP, {"lr": 0.01, "betas": (0.9, 0.95), "eps": 1e-3}, 800),
    (optim.SGDP, {"lr": 0.002, "momentum": 0.91}, 900),
    (optim.AggMo, {"lr": 0.003}, 1800),
    (optim.SWATS, {"lr": 0.1, "amsgrad": True, "nesterov": True}, 900),
    (optim.Adafactor, {"lr": None, "decay_rate": -0.3, "beta1": 0.9}, 800),
    (optim.AdaBelief, {"lr": 1.0}, 5000),
    (optim.Adahessian, {"lr": 1.0, "hessian_power": 1, "seed": 0}, 5000),
    (torch.optim.Adam,{"lr":0.01},5000),
    (torch.optim.SGD,{"lr":0.02},5000),
    (optim.Shampoo,{"lr":0.1},5000),
    (optim.Apollo,{"lr":0.1},5000),
    (optim.MADGRAD, {"lr": 0.02}, 500),
    (optim.LARS, {"lr": 0.002, "momentum": 0.91}, 900),


]

import matplotlib.pyplot as plt

@pytest.mark.parametrize("case", cases, ids=ids)
@pytest.mark.parametrize("optimizer_config", optimizers, ids=ids)
def test_benchmark_function(case, optimizer_config):
    func, initial_state, min_loc = case
    optimizer_class, config, iterations = optimizer_config

    x = torch.Tensor(initial_state).requires_grad_(True)
    x_min = torch.Tensor(min_loc)
    optimizer = optimizer_class([x], **config)
    f_values = []
    for _ in range(iterations):
        optimizer.zero_grad()
        f = func(x)
        f_values.append(f.item())
        f.backward(retain_graph=True, create_graph=True)
        optimizer.step()
    return f_values
    assert torch.allclose(x, x_min, atol=0.001)

    name = optimizer.__class__.__name__
    assert name in optimizer.__repr__()


f_values_adam = test_benchmark_function(cases[0],optimizers[20])
f_values_adabelief = test_benchmark_function(cases[0],optimizers[18])

f_values_adahess = test_benchmark_function(cases[0],optimizers[19])
f_values_sgd = test_benchmark_function(cases[0],optimizers[21])
f_values_shampoo = test_benchmark_function(cases[0],optimizers[22])
f_values_adabound = test_benchmark_function(cases[0],optimizers[8])
f_values_yogi = test_benchmark_function(cases[0],optimizers[9])
f_values_radam = test_benchmark_function(cases[0],optimizers[4])
f_values_padam = test_benchmark_function(cases[0],optimizers[26])
f_values_adan = test_benchmark_function(cases[0],optimizers[27])

# f_values_apollo = test_benchmark_function(cases[0],optimizers[23])

plt.semilogy(f_values_radam, label ='RAdam')
plt.semilogy(f_values_adan, label ='Adan')


plt.semilogy(f_values_adam, label ='Adam')

plt.semilogy(f_values_adahess, label ='AdaHessian')
plt.semilogy(f_values_sgd, label ='SGD')
plt.semilogy(f_values_shampoo, label ='Shampoo')



plt.semilogy(f_values_padam, label ='PAdam')
plt.semilogy(f_values_yogi, label ='Yogi')
plt.semilogy(f_values_adabelief, label ='AdaBelief')
plt.semilogy(f_values_adabound, label ='AdaBound')
plt.semilogy(f_values, label ='PSGD')

# plt.semilogy(f_values_apollo, label ='Apollo')

plt.xlabel('Iterations')
plt.ylabel('Function values')
plt.legend()
