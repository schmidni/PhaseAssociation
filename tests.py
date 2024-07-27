# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

lengths = np.array([195, 182])

# lets create a grid of our two parameters
mu = np.linspace(150, 250)
sigma = np.linspace(0.1, 15)[::-1]

mm, ss = np.meshgrid(mu, sigma)  # just broadcasted parameters


# %%
likelihood = stats.norm(mm, ss).pdf(
    lengths[0]) * stats.norm(mm, ss).pdf(lengths[1])

aspect = mm.max() / ss.max() / 3
extent = [mm.min(), mm.max(), ss.min(), ss.max()]
# extent = left right bottom top

# %%
plt.imshow(likelihood, cmap='Reds', aspect=aspect, extent=extent)
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\sigma$')

# %%
