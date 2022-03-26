import numpy as np
import copy
import matplotlib.pyplot as plt


class Imitation:
    tmax = None
    max_state = None

    def __init__(self, lamda=10, mu=1, nu=5, n=5, number=500):
        self.lamda = lamda  # интенсивность появления новых заявок
        self.mu = mu  # интенсивность обработки заявки
        self.nu = nu  # интенсивность терпеливости заявок в очереди
        self.n = n  # число каналов обработки
        self.num_req = number  # общее число поступивших заявок
        self.takeServe = [{} for _ in range(self.num_req)]  # список взятых в работу заявок
        #           f.e.: {timeStart: - , timeEnd: -, channels:{}}

        self.gone = [0 for _ in range(self.num_req)]  # ушли ли заявки, не дождавшись обработки
        self.queue = []  # очередь заявок
        self.requestsHistory = {}  # сохранение тенденции наличия заявок в очереди

        self.busyChannel = [-1 for _ in range(self.n)]  # заняты ли в текущий момент каналы

        self.t_coming = self.generate_arrive(1 / self.lamda)  # время прихода заявок
        self.t_waiting = self.generate_waiting(1 / self.nu)  # время ухода заявок из очереди без обслуживания
        self.t_service = self.generate_service(1 / self.mu)  # время обслуживания
        self.t_ending = [-1 for _ in range(self.num_req)]  # время окончания обработки заявок

        self.t_coming_start = copy.deepcopy(self.t_coming)

        self.recalcTService = self.set_initial_time_events()  # временные моменты событий
        self.time = self.recalcTService[0]

    def check_queue(self):
        """ Проверяет, обработаны ли все возможные заявки """
        if self.time > self.t_coming_start[-1]:
            busy = 0
            for ch in self.busyChannel:
                if ch != -1:
                    busy += 1
            if busy == 0:
                return False
        return True

    def set_initial_time_events(self):
        times = self.t_coming + self.t_waiting
        times.sort()
        return times

    def generate_arrive(self, lamda):
        """ Генерирует кумулятивный список из распределения Пуассона для времени прибытия заявок """
        distribution = np.random.exponential(scale=lamda, size=self.num_req)
        result = [distribution[0]]

        for index in range(1, distribution.size):
            result.append(result[index - 1] + distribution[index])

        return result

    def generate_waiting(self, lamda):
        """ Генерирует некумулятивный список из распределения Пуассона для времени ухода заявок из очереди """
        distribution = np.random.exponential(scale=lamda, size=self.num_req)
        result = []

        for index in range(0, distribution.size):
            result.append(self.t_coming[index] + distribution[index])

        return result

    def generate_service(self, lamda):
        """ Генерирует некумулятивный список из распределения Пуассона для времени обслуживания """
        distribution = np.random.exponential(scale=lamda, size=self.num_req)
        result = []
        for index in range(0, distribution.size):
            if distribution[index] == 0:
                result.append(0.1)
            else:
                result.append(distribution[index])
        return result

    def request_service_end(self):
        """ Событие окончания обработки заявки.
            Устанавливает время окончания и освобождает занятые каналы,
            при наличии берет в работу заявки из очереди """
        if self.time in self.t_ending:
            req_num = self.t_ending.index(self.time)
            self.t_ending[req_num] = -1

            self.takeServe[req_num]['TimeEnd'] = self.time

            cur_serve = self.takeServe[req_num]
            channels = cur_serve['channels']
            for channel_id in channels:
                self.busyChannel[channel_id] = -1

            if len(self.queue) > 0:
                request = self.queue.pop(0)
                channels = self.reassign_channels()
                self.take_to_work(channels, request)
            else:
                self.help_channels()

            if self.time in self.t_ending:  # если такая заявка не одна
                self.request_service_end()

    def requests_come(self):
        """ Событие поступления новой заявки """
        if self.time in self.t_coming:
            req_num = self.t_coming.index(self.time)
            self.t_coming[req_num] = -1

            channels = self.reassign_channels()
            if len(channels) > 0:  # нашлись свободные
                self.take_to_work(channels, req_num)
            else:
                self.queue.append(req_num)

            if self.time in self.t_coming:
                self.requests_come()

    def take_to_work(self, channels, request):
        """ Событие взятия в работу заявки """
        self.takeServe[request]['TimeStart'] = self.time
        self.takeServe[request]['TimeEnd'] = -1
        self.takeServe[request]['channels'] = channels

        for channel_id in channels:
            self.busyChannel[channel_id] = request
        self.set_end_work_time(request, channels)

    def reassign_channels(self):
        """ Перераспределяет нагрузку на каналы """
        free, busy = self.get_free_busy()
        req_number = len(busy) + 1
        new_engage = self.n // req_number

        if len(free) < new_engage:
            max_busy_index = -1
            max_busy = 0
            for i in busy:
                if len(busy[i]) > max_busy:
                    max_busy = len(busy[i])
                    max_busy_index = i

            new_worklist = []
            if max_busy > new_engage:
                for _ in range(new_engage):
                    new_worklist.append(busy[max_busy_index].pop())
                self.set_end_work_time(max_busy_index, busy[max_busy_index])

            free += new_worklist
        return free

    def set_end_work_time(self, request, channels):
        """ Пересчитывает время окончания обработки заявки """
        if self.takeServe[request]['TimeStart'] == self.time:
            time_end = self.time + (self.t_service[request] / len(channels))
        else:
            time_end = self.time + ((self.t_ending[request] - self.time)
                                    * len(self.takeServe[request]['channels']) / len(channels))
            self.takeServe[request]['channels'] = channels

        self.t_ending[request] = time_end
        self.recalcTService.append(time_end)
        self.recalcTService = list(dict.fromkeys(self.recalcTService))
        self.recalcTService.sort()

    def requests_gone(self):
        """ Событие ухода заявки из очереди без обработки """
        index = 0
        for request in self.queue:
            if self.t_waiting[request] == self.time:
                self.gone[request] = 1
                del self.queue[index]
            else:
                index += 1

    def help_channels(self):
        """ Взаимопомощь другим каналам, когда нет заявок в очереди """
        free, busy = self.get_free_busy()
        req_number = len(busy)

        if (req_number > 0) and (len(free) > 0):
            new_engage = self.n // req_number

            req_index = 0
            for i in busy:
                number_to_add = new_engage - len(busy[i])
                if req_index == (len(busy) - 1):
                    number_to_add = len(free)

                for _ in range(number_to_add):
                    if len(free) > 0:
                        channel = free.pop()
                        busy[i].append(channel)
                        self.busyChannel[channel] = i
                    else:
                        raise IndexError('Свободные каналы: ', free, 'пытаемся добавить к каналам ', busy[i])

                self.set_end_work_time(i, busy[i])
                req_index += 1

    def get_free_busy(self):
        free = []
        busy = {}
        for c in range(self.n):
            if self.busyChannel[c] == -1:
                free.append(c)
            else:
                if self.busyChannel[c] in busy:
                    busy[self.busyChannel[c]].append(c)
                else:
                    busy[self.busyChannel[c]] = [c]
        return free, busy

    def check_next_event(self):
        """ Проверка на следующее событие и установка следующего времени остановки """
        index = self.recalcTService.index(self.time)
        if index < (len(self.recalcTService) - 1):
            self.time = self.recalcTService[index + 1]

    def set_history(self):
        """ Сохранение количества заявок в системе """
        req_in_work = []
        for ch in self.busyChannel:
            if (ch != -1) and (ch not in req_in_work):
                req_in_work.append(ch)
        self.requestsHistory[round(self.time, 2)] = len(self.queue) + len(req_in_work)

    def process_queue(self):
        """ Процесс обработки очереди """
        while self.check_queue():
            self.request_service_end()
            self.requests_come()
            self.requests_gone()

            self.set_history()
            self.check_next_event()

    def print_plot_workflow(self):
        fig, ax = plt.subplots()
        ax.grid()

        for req in range(len(self.takeServe)):
            if self.gone[req] == 1:
                plt.barh(req, (self.t_waiting[req] - self.t_coming_start[req]),
                         left=self.t_coming_start[req], color='r')
            else:
                plt.barh(req, (self.takeServe[req]['TimeStart'] - self.t_coming_start[req]),
                         left=self.t_coming_start[req], color='yellow')
                plt.barh(req, (self.takeServe[req]['TimeEnd'] - self.takeServe[req]['TimeStart']),
                         left=self.takeServe[req]['TimeStart'], color='b')

        plt.rcParams["figure.figsize"] = (self.time + 50, self.n)
        plt.show()

    def print_main_params(self):
        print('t_coming', self.t_coming_start)
        print('t_service', self.t_service)
        print('t_waiting', self.t_waiting)
        print('takeServe', self.takeServe)
        print('history', self.requestsHistory)

    @staticmethod
    def get_frequency_char(c_states, c_char, t_moments):
        t_index = 0
        for t in t_moments:
            t_time = round(t, 2)
            for index_state in range(len(c_char)):
                state = c_char[index_state]
                c_states[state[t_time]][t_index] += 1
            t_index += 1

        count_runs = len(c_char)
        t_index = 0
        for _ in t_moments:
            for state in c_states:
                c_states[state][t_index] /= count_runs
            t_index += 1

        return c_states

    @staticmethod
    def draw_frequency_characteristics(chars, t_moments):
        fig, ax = plt.subplots()
        for sys_state in chars:
            plt.plot(t_moments, chars[sys_state], linewidth=1, label='state ' + str(sys_state))

        plt.title("График частотных характеристик СМО")
        plt.grid()
        plt.legend()
        plt.show()

    @classmethod
    def run(cls, samples=1000, lamda=10, mu=1, nu=5, n=5, number=500):
        count_char = []
        minmax_time = np.inf
        run_number = samples

        for _ in range(run_number):
            request_poll = cls(lamda, mu, nu, n, number)
            request_poll.process_queue()
            count_char.append(request_poll.requestsHistory)
            current_max_time = round(request_poll.time, 2)

            if current_max_time < minmax_time:
                minmax_time = current_max_time

        intervals = list(np.arange(0.00, minmax_time, 0.01))
        count_time_moments = len(intervals)
        step = 0.01
        cls.tmax = count_time_moments * step

        avail_states = {}
        for time in intervals:
            time = round(time, 2)
            for run_index in range(len(count_char)):
                if time not in count_char[run_index]:
                    if time == 0:
                        count_char[run_index][time] = 0
                    else:
                        count_char[run_index][time] = count_char[run_index][round(time - step, 2)]

                if count_char[run_index][time] not in avail_states:
                    avail_states[count_char[run_index][time]] = 0

        cls.max_state = max(avail_states, key=lambda x: x)
        states = {}
        for st in range(cls.max_state + 1):
            states[st] = [0 for _ in range(count_time_moments)]

        frequency_characteristic = cls.get_frequency_char(states, count_char, intervals)
        cls.draw_frequency_characteristics(frequency_characteristic, intervals)
