import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# later

mius = [55.43333333, 56.36666667, 50.2, 45, 55.56666667]
sds = [2.77578719, 2.41459917, 1.66162024, 0, 1.58665064]

# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(30, 70, 0.01)

plt.plot(x_axis, norm.pdf(x_axis,mius[0],sds[0]))
plt.show()