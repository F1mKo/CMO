import csv
import numpy as np
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


def get_objective():
    def f(a, k, n_tot):
        return service_price * a * (1 - taxes) - service_cost * (n_tot - k)

    n_var = [i for i in range(1, 10)]
    f_list = []
    model_stats = []
    service_price = 100
    service_cost = 30
    taxes = 0.1
    max = 0
    n_opt = 0
    for n_i in n_var:
        prob = StatMod(lamda, mu, nu, n_i, num_req, 15, 10)
        prob.run()
        f_list.append(f(prob.A, prob.n_w, n_i))
        model_stats.append((n_i, f_list[-1], np.round(prob.nq_prob, 3), np.round(prob.rej_prob, 3), np.round(prob.Q, 3), np.round(prob.A, 3), np.round(prob.n_w, 3)))
        if f_list[-1] > max:
            max = f_list[-1]
            n_opt = n_i

    ax = plt.axes()
    plt.plot(n_var, f_list, color='r', label='F(n)')
    plt.plot(n_opt, max, 'bo')
    plt.title('Зависимость ЦФ от количества каналов СМО')
    ax.set_xlabel('Количество каналов n')
    ax.set_ylabel('Значение ЦФ')
    plt.grid()
    plt.legend()
    plt.show()
    # Write to csv
    Columns = ['n', 'F', 'p_0', 'P_rej', 'Q', 'A', 'k']
    with open('model_out.csv', 'w') as my_file:
        wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
        wr.writerow(Columns)
        wr.writerows(model_stats)
    return n


samples = 1000  # количество запусков имитационной модели
lamda = 8  # интенсивность появления новых заявок
mu = 2  # интенсивность обработки заявки
nu = 3  # интенсивность терпеливости заявок в очереди
n = 5  # число каналов обработки
num_req = 100  # общее число поступивших заявок

#Imitation.run(samples, lamda, mu, nu, n, num_req)
#prob = StatMod(lamda, mu, nu, n, num_req, Imitation.max_state + 1, Imitation.tmax)
#prob.run()

get_objective()
