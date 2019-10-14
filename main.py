import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from estimation import simulated_mm
"""
This program -hopefully- estimates the household's
consumption-savings problem with habit in consumption.

@author: Davide Coluccia
"""
#################################
#   INITIAL GUESSES
#################################
alpha0 = 0.1
sigma0 = 0.1
data = pd.read_csv('simulated_data.csv')
#################################
#   RUN ESTIMATION
#################################
est_store  = simulated_mm(sigma0, alpha0, data)
x = est_store.alpha_grid
y = est_store.sigma_grid
z = est_store.outcome_function
#################################
#   PLOT RESULTS
#################################
# Two parameters
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')
yy, xx = np.meshgrid(y, x)
ax.plot_surface(yy, xx, z, cmap=plt.cm.viridis, linewidth=0.2)

ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'GMM Loss')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
yy, xx = np.meshgrid(y, x)
ax.plot_surface(yy, xx, np.log(z), cmap=plt.cm.viridis, linewidth=0.2)

ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$\alpha$')
ax.set_zlabel(r'GMM $\log$ Loss')
plt.show()

## One parameter
#fig, ax =  plt.subplots(2, 1)
#fig.suptitle(r'GMM estimation: $\mathcal{A} = \{0.0,\dots,0.3\}$')
## linear
#ax[0].plot(x, z)
#ax[0].set_yscale('linear')
##ax[0].set_xlabel(r'$\alpha$')
#ax[0].set_ylabel(r'$\mu^T \mathbf{I} \mu$')
#ax[0].grid()
#
## log
#ax[1].plot(x, z, 'r')
#ax[1].set_yscale('log')
#ax[1].set_xlabel(r'$\alpha$')
#ax[1].set_ylabel(r'$\log ( \mu^T \mathbf{I} \mu  )$')
#ax[1].grid()
#
#plt.show()
#################################
#   TABLE RESULTS
#################################
#est_coefs = est_store.est_coef
#print(est_coefs)