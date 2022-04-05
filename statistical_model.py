import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

class StatMod:
    def __init__(self, lamda=10, mu=1, nu=5, n=5, num_req=50, imitation_states=50, tmax=5):
        self.lamda = lamda  # интенсивность появления новых заявок
        self.mu = mu  # интенсивность обработки заявки
        self.nu = nu  # интенсивность терпеливости заявок в очереди
        self.n = n  # число каналов обработки
        self.num_req = num_req  # общее число поступивших заявок (максимально возможное число состояний)
        self.max_states = imitation_states
        self.tmax = tmax  # максимально допустимый момент времени
        self.ts = []
        self.ys = []
        self.y0, self.t0 = [1] + [0 for _ in range(1, self.num_req + 1)], 0  # начальные условия
        self.st_names = [name for name in range(self.max_states)]
        self.ps = [[] for _ in range(self.max_states)]
        self.Y = np.array(0)
        self.tau = 0.0025  # шаг интегрирования

    def foot(self, t, y):
        # обработчик шага
        self.ts.append(t)
        self.ys.append(list(y.copy()))

        for state in range(self.max_states):
            self.ps[state].append(y[state])
        if t > self.tmax:
            return -1

    def solve(self):
        """ Численное интегирование СДУ и последующая запись результатов"""
        ODE = ode(self.f)
        ODE.set_integrator('dopri5', max_step=0.05)
        ODE.set_solout(self.foot)
        ODE.set_initial_value(self.y0, self.t0)  # задание начальных значений
        ODE.integrate(self.tmax)  # решение ОДУ

    def f(self, t, p):
        """ Функция правых частей системы ОДУ """
        n, num_req, lamda, mu, nu = self.n, self.num_req, self.lamda, self.mu, self.nu
        result = [-lamda * p[0] + n * mu * p[1]]
        result += [(lamda * p[i - 1] - (lamda + n * mu) * p[i] + n * mu * p[i + 1])
                   for i in range(1, n)]

        result += [lamda * p[i - 1] - (lamda + n * mu + (i - n) * nu) * p[i] +
                   (n * mu + (i - n + 1) * nu) * p[i + 1] for i in range(n, num_req)]

        result += [lamda * p[num_req - 1] - (lamda + n * mu + (num_req - n) * nu) * p[num_req]]

        return result

    def get_report(self):
        """ Построение графика вероятностей для состояний системы """
        fig1, ax1 = plt.subplots()
        self.Y = np.array(self.ys)

        for sys_state in range(len(self.ps)):
            plt.plot(self.ts, self.ps[sys_state], linewidth=1, label='state ' + str(self.st_names[sys_state]))

        print("Предельные значения распределения: ", self.Y[-1])
        # print("Сумма вероятностей: ", sum(self.Y[-1]))

        plt.title("График вероятностей состояний СМО")
        plt.grid()
        plt.legend()
        plt.show()

    def increment(self, t, y, h):
        """ Вычисление коэффициентов для 4-х этапного метода Рунге-Кутты """
        k1 = self.mult(h, self.f(t, y))
        k2 = self.mult(h, self.f(t, self.add(y, self.mult(h / 2.0, k1))))
        k3 = self.mult(h, self.f(t, self.add(y, self.mult(h / 2.0, k2))))
        k4 = self.mult(h, self.f(t, self.add(y, self.mult(h, k3))))

        result = self.add(self.mult(1 / 6, k1), self.mult(1 / 3, k2))
        result = self.add(result, self.mult(1 / 3, k3))
        result = self.add(result, self.mult(1 / 6, k4))

        return result

    def runge_kutta(self):
        """ Численное интегирование СДУ и последующая запись результатов"""
        self.ts.append(self.t0)  # внесение начальных значений
        self.ys.append(self.y0)

        cur_t = self.t0
        cur_y = self.y0
        tau = self.tau
        rtol = 10 ** (-15)

        for state in range(self.max_states):
            self.ps[state].append(cur_y[state])

        while cur_t < self.tmax:  # цикл по временному промежутку интегрирования
            cur_y = self.add(cur_y, self.increment(cur_t, cur_y, tau))  # расчёт значения в точке t0,y0 для задачи Коши
            cur_y2 = self.add(cur_y, self.increment(cur_t, cur_y, tau / 2.0))  # расчёт значения в точке t0,y0 для задачи Коши
            epsilon = (abs(np.array(cur_y) - np.array(cur_y2))).max()
            if epsilon > rtol * 15:
                tau = tau / 2.0
            else:
                if epsilon >= rtol / 32:
                    cur_t = cur_t + tau / 2.0
                    cur_y = cur_y2
                    self.ts.append(cur_t)
                    self.ys.append(cur_y)
                    for state in range(self.max_states):
                        self.ps[state].append(cur_y[state])
                else:
                    cur_t = cur_t + tau / 2.0
                    cur_y = cur_y2
                    self.ts.append(cur_t)
                    self.ys.append(cur_y)
                    if tau * 2 <= 0.00001:
                        tau = tau * 2
                    for state in range(self.max_states):
                        self.ps[state].append(cur_y[state])

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
        # print('Сравнение предельных вероятностей:')
        max_variate = 0
        for sys_state in range(self.max_states + 1):
            if self.Y[-1][sys_state] - ultimate_p[sys_state] > max_variate:
                max_variate = self.Y[-1][sys_state] - ultimate_p[sys_state]
            # print('state ' + str(sys_state), self.Y[-1][sys_state] - ultimate_p[sys_state])
        print('Наибольшее отклонение предельных вероятностей:', max_variate)

    def calc_metrics(self):
        # расчет предельных вероятностей состояний системы:
        p0 = self.Y[-1][0]
        rho = self.lamda / (self.n * self.mu)
        nq_prob = self.arr_sum(self.Y[-1], 0, self.n + 1, 0)
        n_s = self.arr_sum(self.Y[-1], 1, self.n + 1, 1)
        n_t = self.arr_sum(self.Y[-1], 1, self.max_states + 1, 1)
        n_w = n_t - n_s
        A = self.lamda - self.nu * n_w
        Q = A / self.lamda
        rej_prob = 1 - self.mu / self.lamda * n_t
        print('Статистическая модель -', 'Интенсивность нагрузки системы:', rho)
        print('Статистическая модель -', 'Вероятность простоя системы:', p0)
        print('Статистическая модель -', 'Вероятность отсутствия очереди:', nq_prob)
        print('Статистическая модель -', 'Среднее число заявок под обслуживанием:', n_s)
        print('Статистическая модель -', 'Среднее число заявок в системе:', n_t)
        print('Статистическая модель -', 'Среднее число заявок в очереди:', n_w)
        print('Статистическая модель -', 'Абсолютная пропускная способность:', A)
        print('Статистическая модель -', 'Относительная пропускная способность:', Q)
        print('Статистическая модель -', 'Вероятность отказа:', rej_prob)

    @staticmethod
    def mult(element, array):
        # переопределение умножения для массивов
        for i in range(len(array)):
            array[i] *= element

        return array

    @staticmethod
    def add(array, array1):
        # переопределение сложения для массивов
        for i in range(len(array)):
            array[i] += array1[i]

        return array

    @staticmethod
    def arr_sum(array, n_from, n_to, degree):
        # сумма элементов массива
        total = 0
        for i in range(n_from, n_to):
            if (i != 0) or (degree != 0):
                total += array[i] * (i ** degree)
            else:
                total += array[i]
        return total

    def run(self):
        """ Основная функция запуска стасистической модели"""
        self.runge_kutta()
#        self.solve()
        self.get_report()
#        self.calc_lim_prob()
        self.calc_metrics()
