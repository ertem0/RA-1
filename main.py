import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import norm
from math import comb
from time import time
from scipy.interpolate import interp1d

start = time()
n = 1000  # Number of levels
N = 10  # Number of balls

def drop_ball(n):
    pos = [0, 0]
    for i in range(n):
        if random.randint(0, 1):
            pos[0] += 1
        else:
            pos[1] += 1
    return pos

x_fine = np.linspace(0, n, 500)
x = np.arange(0, n + 1)

binomial_probabilities = [comb(n, i) * (0.5 ** n) for i in range(n + 1)]

experiment = [drop_ball(n) for _ in range(N)]
normalized_pos = [pos[1] for pos in experiment]
experiment_probs = [normalized_pos.count(i)/N for i in x]

mu = n / 2  
sigma = np.sqrt(n / 4) 
normal = norm.pdf(x, mu, sigma)
normal_approx = norm.pdf(x_fine, mu, sigma)

squared_errors = (normal - experiment_probs) ** 2
mqe = np.mean(squared_errors)

print(mqe)

print(time() - start)

plt.bar(x, experiment_probs, color='green', label='Experimental Distribution', zorder=1)

plt.plot(x_fine, normal_approx, color='red', lw=2, label='Normal Approximation', zorder=3)

plt.scatter(x, binomial_probabilities, color='blue', label='Binomial Distribution', zorder=2)

plt.xlabel(f'Position (i)')
plt.ylabel(f'P[X=i]')
plt.title(f'(levels={n}, balls={N})\nmqe={round(mqe,8)}')
plt.legend()
plt.savefig(f"level{n}_balls{N}.png")
plt.show()
