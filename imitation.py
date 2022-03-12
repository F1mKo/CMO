import numpy as np


class RequestPoll:
    def __init__(self, lamda, mu, nu, n, number):
        self.lamda = lamda  # интенсивность появления новых заявок
        self.mu = mu
        self.nu = nu
        self.n = n  # число каналов обработки
        self.num_req = number  # общее число поступивших заявок
        self.takeServe = [{} for _ in range(self.num_req)]  # список взятых в работу заявок
        # f.e.: {timeStart: - , timeEnd: -, channels:{}}

        self.gone = [0 for _ in range(self.num_req)]  # ушли ли заявки, не дождавшись обработки
        self.queue = []  # очередь заявок
        self.queueHistory = []  # сохранение тенденции наличия заявок в очереди

        self.busyChannel = [0 for _ in range(self.n)]  # заняты ли в текущий момент каналы
        self.workChannelsInfo = [[] for _ in range(self.n)]  # информация о работе каналов
        # f.e.: [[{request: -, timeStart: -, timeEnd: -}, {}], [], []]

        self.t_coming = self.generate_arrive(self.lamda)  # время прихода заявок
        self.t_waiting = self.generate_waiting(self.nu)  # время ухода заявок из очереди без обслуживания
        self.t_service = self.generate_service(self.mu)  # время обслуживания

        self.recalcTService = []  # временные моменты событий
        self.time = 0
        # self.timeLaps = np.arange(0.0, self.t_coming[-1], 0.1).tolist()

        # self.process_queue()

    def check_queue(self):
        """ Проверяет, обработаны ли все возможные заявки """
        # if len(self.takeServe) < self.num_req:
        if self.time <= self.t_coming[-1]:
            return True
        else:
            return False

    def generate_arrive(self, lamda):
        """ Генерирует кумулятивный список из распределения Пуассона для времени прибытия заявок """
        distribution = np.random.poisson(lam=lamda, size=self.num_req)
        result = [distribution[0]]

        for index in range(1, distribution.size):
            result.append(result[index - 1] + distribution[index])

        return result

    def generate_waiting(self, lamda):
        """ Генерирует кумулятивный список из распределения Пуассона для времени ухода заявок из очереди """
        distribution = np.random.poisson(lam=lamda, size=self.num_req)
        result = []

        for index in range(0, distribution.size):
            result.append(self.t_coming[index] + distribution[index])

        return result

    def generate_service(self, lamda):
        """ Генерирует некумулятивный список из распределения Пуассона для времени обслуживания """
        distribution = np.random.poisson(lam=lamda, size=self.num_req)
        result = []
        for index in range(0, distribution.size):
            if distribution[index] == 0:
                result.append(0.1)
            else:
                result.append(distribution[index])
        return result

    def request_service_end(self):
        """ Событие окончания обработки заявки
            Устанавливает время окончания
            и освобождает занятые каналы,
            при наличии берет в работу заявки из очереди """
        if self.time in self.recalcTService:
            req_num = self.recalcTService.index(self.time)

            self.recalcTService[req_num] = -1
            self.takeServe[req_num]['TimeEnd'] = self.time

            cur_serve = self.takeServe[req_num]
            channels = cur_serve['channels']
            for channel in channels:
                self.busyChannel[channel] = 0

            # ToDo: брать в работу новую заявку

            if self.time in self.recalcTService:  # если такая заявка не одна
                self.request_service_end()

    def requests_come(self):
        """ Событие поступления новой заявки """
        if self.time in self.t_coming.index:
            req_num = self.t_coming.index(self.time)



            if self.time in self.t_coming.index:
                self.requests_come()

    def requests_gone(self):
        """ Событие ухода заявки из очереди без обработки """
        index = 0
        req_to_del = []
        for request in self.queue:
            index += 1
            if self.t_waiting[request] == self.time:
                req_to_del.append(index)
                self.gone[request] = 1

        for index in req_to_del:
            del self.queue[index]

    def check_next_event(self):
        """ Проверка на следующее событие и установка следующего времени остановки """
        # ToDo: реализовать по-человечески
        self.recalcTService = self.t_coming

    def process_queue(self):
        """ Процесс обработки очереди """
        while self.check_queue():
            self.request_service_end()
            self.requests_come()
            self.requests_gone()

            self.queueHistory.append(len(self.queue))


request_poll = RequestPoll(10, 3, 4, 5, 50)

print('t_coming', request_poll.t_coming)
print('t_service', request_poll.t_service)
print('t_waiting', request_poll.t_waiting)
