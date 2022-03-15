import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode


class StatMod:
    def __init__(self, lamda=10, mu=1, nu=5, n=5, num_req=50, imitation_states=50):
        self.lamda = lamda  # интенсивность появления новых заявок
        self.mu = mu  # интенсивность обработки заявки
        self.nu = nu  # интенсивность терпеливости заявок в очереди
        self.n = n  # число каналов обработки
        self.num_req = num_req  # общее число поступивших заявок (максимальное возможное число состояний)
        self.max_states = imitation_states

    def f(self, t, p):
        # функция правых частей системы ОДУ
        lamda = self.lamda
        mu = self.mu  # интенсивность обработки заявки
        nu = self.nu  # интенсивность терпеливости заявок в очереди
        n = self.n  # число каналов обработки
        num_req = self.num_req  # общее число поступивших заявок (максимальное возможное число состояний)
        return [-lamda * p[0] + n * mu * p[1]] \
               + [(lamda * p[i - 1] - (lamda + n * mu) * p[i] + n * mu * p[i + 1]) for i in range(1, n)] \
               + [lamda * p[i - 1] - (lamda + n * mu + (i - n) * nu) * p[i] + (n * mu + (i - n + 1) * nu) * p[i + 1]
                  for i in range(n, num_req)] \
               + [lamda * p[num_req - 1] - (lamda + n * mu + (num_req - n) * nu) * p[num_req]]

    def foot(self, t, y):
        # обработчик шага
        self.ts.append(t)
        self.ys.append(list(y.copy()))

        for state in range(self.max_states):
            self.ps[state].append(y[state])
        # for state in range(self.n + 3):
        #    self.ps[state].append(y[state])
        # index = self.n + 3
        # for state in range(self.n + 3, self.num_req, 5):
        #    self.ps[index].append(y[state])
        #    index += 1
        if t > self.tmax:
            return -1

    def get_report(self):
        # построение графика вероятностей для состояний системы
        fig1, ax1 = plt.subplots()
        # fig1.set_facecolor('white')
        self.Y = np.array(self.ys)

        for sys_state in range(len(self.ps)):
            plt.plot(self.ts, self.ps[sys_state], linewidth=1, label='state ' + str(self.st_names[sys_state]))

        print("Предельные значения распределения: ", self.Y[-1])
        print("Сумма вероятностей: ", sum(self.Y[-1]))

        plt.title("График вероятностей состояний СМО")
        plt.grid()
        plt.legend()
        plt.show()

        self.calc_lim_prob()

    def solve(self):
        # Численное интегирование СДУ и последующая запись результатов
        self.ts = []
        self.ys = []
        self.ps = [[] for _ in range(self.max_states)]
        self.st_names = [name for name in range(self.max_states)]
        # self.ps = [[] for _ in range(self.n + 3)] + [[] for _ in range(self.n + 3, self.num_req, 5)]
        # self.st_names = [name for name in range(self.n + 3)] + [name for name in range(self.n + 3, self.num_req, 5)]

        self.tmax = 10  # максимально допустимый момент времени
        self.y0, self.t0 = [1] + [0 for _ in range(1, self.num_req + 1)], 0  # начальные условия

        ODE = ode(self.f)
        ODE.set_integrator('dopri5', max_step=0.05)
        ODE.set_solout(self.foot)
        ODE.set_initial_value(self.y0, self.t0)  # задание начальных значений
        ODE.integrate(self.tmax)  # решение ОДУ

    def calc_lim_prob(self):
        # расчет предельных вероятностей состояний системы:
        p0 = 0
        product = [1]
        for sys_state in range(1, self.num_req + 1):
            prod_state = 1
            for j in range(1, sys_state + 1):
                prod_state *= (self.n * self.mu + j * self.nu)
            product.append(prod_state)

        for channel in range(self.n + 1):
            p0 += (self.lamda / (self.n * self.mu)) ** channel

        for sys_state in range(1, self.num_req + 1):
            p0 += (self.lamda ** (self.n + sys_state)) / (((self.n * self.mu) ** self.n) * product[sys_state])
        p0 = p0 ** (-1)

        ultimate_p = [p0]
        for sys_state in range(1, self.n + 1):
            ultimate_p.append(p0 * (self.lamda / (self.n * self.mu)) ** sys_state)

        for sys_state in range(self.n + 1, self.num_req + 1):
            prod_state = 1
            for j in range(1, sys_state - self.n + 1):
                prod_state *= (self.n * self.mu + j * self.nu)
            ultimate_p.append(p0 * ((self.lamda ** sys_state) / (((self.n * self.mu) ** self.n) * prod_state)))

        print(ultimate_p)
        print('Сравнение предельных вероятностей:')
        for sys_state in range(self.num_req + 1):
            print('state ' + str(sys_state), self.Y[-1][sys_state] - ultimate_p[sys_state])

    def run(self):
        # Основная функция запуска стасистической модели
        self.solve()
        self.get_report()
        self.calc_lim_prob()

# prob = StatMod()
# prob.run()
