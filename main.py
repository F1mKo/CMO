import matplotlib.pyplot as plt
import seaborn as sns
from imitation_model import Imitation
from statistical_model import StatMod


large = 34
med = 22
small = 16
params = {'legend.fontsize': small,
          'figure.figsize': (8, 9),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}

plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

samples = 10
lamda = 10  # интенсивность появления новых заявок
mu = 1  # интенсивность обработки заявки
nu = 5  # интенсивность терпеливости заявок в очереди
n = 5  # число каналов обработки
num_req = 50  # общее число поступивших заявок

Imitation.run(samples, lamda, mu, nu, n, num_req)
prob = StatMod(lamda, mu, nu, n, num_req, Imitation.max_state + 1, Imitation.tmax)
prob.run()
