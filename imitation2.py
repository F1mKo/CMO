import numpy as np


class RequestPoll:
    def __init__(self, lamda, n, number, multipliers):
        self.lamda = lamda  # интенсивность появления новых заявок
        self.n = n  # число каналов обработки
        self.num_req = number  # общее число поступивших заявок
        self.takeServe = [{} for _ in range(self.num_req)]  # список взятых в работу заявок
        # f.e.: {timeStart: - , timeEnd: -, channels:{}}
        self.gone = np.zeros(self.num_req)  # ушли ли заявки, не дождавшись обработки
        self.queue = []  # очередь заявок
        self.busyChannel = np.zeros(self.n)  # заняты ли в текущий момент каналы
        self.workChannelsInfo = [[] for _ in range(self.n)]  # информация о работе каналов
        # f.e.: [[{request: -, timeStart: -, timeEnd: -}, {}], [], []]

        self.t_coming = self.generate_arrive(self.lamda * multipliers['come'])  # время прихода заявок
        self.t_service = self.generate_distribution(self.lamda * multipliers['serve'])  # время обслуживания
        self.t_waiting = self.generate_distribution(self.lamda * multipliers['wait'])  # max время ожидания заявок

        self.recalcTService = self.t_service.copy()  # пересчитанное время обработки заявки (-1 -> обработаны)
        self.time = 0

        self.process_queue()

    def check_queue(self):
        """ Проверяет, обработаны ли все возможные заявки """
        # if len(self.takeServe) < self.num_req:
        if self.time <= self.t_coming[-1]:
            return True
        else:
            return False

    def generate_arrive(self, lamda):
        """ Генерирует кумулятивный список из распределения Пуассона для времени прибытия заявок """
        result = np.zeros(self.num_req)
        distribution = np.random.poisson(lam=lamda, size=self.num_req)
        result[0] = distribution[0]

        for index in range(1, result.size):
            result[index] = result[index - 1] + distribution[index]

        return result

    def generate_distribution(self, lamda):
        """ Генерирует некумулятивный список из распределения Пуассона """
        result = np.random.poisson(lam=lamda, size=self.num_req)
        return result

    def request_service_end(self):
        """ Проверяет завершилась ли обработка заявок
            Если завершилась, устанавливает время окончания
            и освобождает занятые каналы """
        req_num = self.recalcTService.index(self.time)
        while req_num:
            self.recalcTService[req_num] = -1
            cur_serve = self.takeServe[req_num]
            channels = cur_serve['channels']

    def requests_gone(self):
        """ Проверяет ушли ли заявки не дождавшись обработки """
        return self.time

    def requests_come(self):
        """ Проверяет пришли ли новые заявки """
        return self.time

    def requests_take_to_serve(self):
        """ Проверяет есть ли заявки, которые можно взять в работу и свободные каналы обработки """
        return self.time

    def process_queue(self):
        """ Процесс обработки очереди """
        while self.check_queue():
            self.request_service_end()
            self.requests_gone()
            self.requests_come()
            self.requests_take_to_serve()


time_multipliers = {'come': 1, 'serve': 1.5, 'wait': 2}
request_poll = RequestPoll(10, 5, 50, time_multipliers)

print('t_coming', request_poll.t_coming)
print('t_service', request_poll.t_service)
print('t_waiting', request_poll.t_waiting)
