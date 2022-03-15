import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import ode
from imitation_model import Imitation
from statistical_model import Stat_mod

large = 34; med = 22; small = 16
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

Imitation.run(1000, 10, 1, 5, 5, 50)
prob = Stat_mod()
prob.run()
