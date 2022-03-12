import random
import numpy as np


def print_all(object_info):
    print(object_info)


class RequestPoll:
    def __init__(self):
        self.lamda = 10  # интенсивность появления новых заявок
        self.n = 5  # число каналов обработки
        self.num_req = 50  # общее число поступивших заявок
        self.t_coming = self.generate_distribution(1)
        self.t_service = self.generate_distribution(2)
        self.t_waiting = self.generate_distribution(3)

        # self.mu = 1  # скорость обработки заявки
        # self.nu = 5  # ожидание

    def generate_distribution(self, multiplier):
        result = np.zeros(self.num_req)
        distribution = np.random.poisson(lam=self.lamda, size=self.num_req)
        result[0] = int(distribution[0]) * multiplier

        for index in range(1, result.size):
            result[index] = result[index - 1] + distribution[index] * multiplier

        return result


request_poll = RequestPoll()

print('t_coming', request_poll.t_coming)
print('t_service', request_poll.t_service)
print('t_waiting', request_poll.t_waiting)
