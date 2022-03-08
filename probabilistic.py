import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


# функция правых частей системы ОДУ
def f(t, p):
    return [-lamda * p[0] + n * mu * p[1]] \
           + [(lamda * p[i - 1] - (lamda + n * mu) * p[i] + n * mu * p[i + 1]) for i in range(1, n)] \
           + [lamda * p[i - 1] - (lamda + n * mu + (i - n) * nu) * p[i] + (n * mu + (i - n + 1) * nu) * p[i + 1]
              for i in range(n, num_req)] \
           + [lamda * p[num_req - 1] - (lamda + n * mu + (num_req - n) * nu) * p[num_req]]


def foot(t, y):  # обработчик шага
    ts.append(t)
    ys.append(list(y.copy()))
    for state in range(n + 3):
        ps[state].append(y[state])
    index = n + 3
    for state in range(n + 3, num_req, 5):
        ps[index].append(y[state])
        index += 1
    if t > tmax:
        return -1


lamda = 10  # интенсивность появления новых заявок
n = 5  # число каналов обработки
num_req = 50  # общее число поступивших заявок
mu = 1  # скорость обработки заявки
nu = 5  # ожидание

ts = []
ys = []
ps = [[] for _ in range(n + 3)] + [[] for _ in range(n + 3, num_req, 5)]
st_names = [name for name in range(n + 3)] + [name for name in range(n + 3, num_req, 5)]

tmax = 10  # максимально допустимый момент времени
y0, t0 = [1] + [0 for _ in range(1, num_req + 1)], 0  # начальные условия
ODE = ode(f)
ODE.set_integrator('dopri5', max_step=0.05)
ODE.set_solout(foot)
fig, ax = plt.subplots()
fig.set_facecolor('white')

ODE.set_initial_value(y0, t0)  # задание начальных значений
ODE.integrate(tmax)  # решение ОДУ
Y = np.array(ys)

for sys_state in range(len(ps)):
    plt.plot(ts, ps[sys_state], linewidth=1, label='state ' + str(st_names[sys_state]))

print("Предельные значения распределения: ", Y[-1])
print("Сумма вероятностей: ", sum(Y[-1]))

plt.grid()
plt.legend()
plt.show()

# расчет предельных вероятностей:
p0 = 0
product = [1]
for sys_state in range(1, num_req + 1):
    prod_state = 1
    for j in range(1, sys_state + 1):
        prod_state *= (n * mu + j * nu)
    product.append(prod_state)

for channel in range(n + 1):
    p0 += (lamda / (n * mu)) ** channel

for sys_state in range(1, num_req + 1):
    p0 += (lamda ** (n + sys_state)) / (((n * mu) ** n) * product[sys_state])
p0 = p0 ** (-1)

ultimate_p = [p0]
for sys_state in range(1, n + 1):
    ultimate_p.append(p0 * (lamda / (n * mu)) ** sys_state)

for sys_state in range(n + 1, num_req + 1):
    prod_state = 1
    for j in range(1, sys_state - n + 1):
        prod_state *= (n * mu + j * nu)
    ultimate_p.append(p0 * ((lamda ** sys_state) / (((n * mu) ** n) * prod_state)))

print(ultimate_p)
print('Сравнение предельных вероятностей:')
for sys_state in range(num_req + 1):
    print('state ' + str(sys_state), Y[-1][sys_state] - ultimate_p[sys_state])




