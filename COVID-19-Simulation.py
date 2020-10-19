import matplotlib.pyplot as plt
import numpy as np


class Hospital(object):
    def __init__(self, width=50, height=20):
        self.width = width
        self.height = height
        self.count = width*height
        self.beds = np.ones([width*height, 2])
        self.beds[:, 0] = np.tile(np.arange(400, 400+width), height)
        self.beds[:, 1] = np.repeat(np.arange(-200, -200+height), width)


class People(object):
    def __init__(self, count=1000, first_infected_count=3, hospital=Hospital()):
        self.count = count
        self.first_infected_count = first_infected_count
        self.hospital = hospital
        self.init()

    def init(self):
        self._people = np.random.normal(0, 100, (self.count, 2))
        self.reset()

    def reset(self):
        self._round = 0
        self._status = np.array([0] * self.count)
        self._timer = np.array([0] * self.count)
        self.random_people_state(self.first_infected_count, 1)

    def random_people_state(self, num, state=1):
        """随机挑选人设置状态//Select people to be initially infected at random
        """
        assert self.count > num
        # TODO：极端情况下会出现无限循环//In extreme cases, infinite loop would appear
        n = 0
        while n < num:
            i = np.random.randint(0, self.count)
            if self._status[i] == state:
                continue
            else:
                self.set_state(i, state)
                n += 1

    def set_state(self, i, state):
        self._status[i] = state
        # 记录状态改变的时间//Record the time of status change
        self._timer[i] = self._round

    def random_movement(self, width=1):
        """随机生成移动距离//Generate moving distance at random

        :param width: 控制距离范围//Control the movement range
        :return:
        """
        return np.random.normal(0, width, (self.count, 2))

    def random_switch(self, x=0.):
        """随机生成开关，0 - 关，1 - 开//Generate random switch, 0 for turn off, 1 for turn on

        x 大致取值范围 -1.99 - 1.99；//x represents the intention of people for mobility
        对应正态分布的概率， 取值 0 的时候对应概率是 50%//Obey a Gaussian distribution of N(0,1)
        :param x: 控制开关比例
        :return:
        """
        normal = np.random.normal(0, 1, self.count)
        switch = np.where(normal < x, 1, 0)
        return switch

    @property
    def healthy(self):
        return self._people[self._status == 0]

    @property
    def infected(self):
        return self._people[self._status == 1]

    @property
    def confirmed(self):
        return self._people[self._status == 2]

    @property
    def isolated(self):
        return self._people[self._status == 3]

    def move(self, width=1, x=.0):
        movement = self.random_movement(width=width)
        # 限定特定状态的人员移动//Constrain the movements of people with special status
        switch = self.random_switch(x=x)
        movement[(self._status == 3) | switch == 0] = 0
        # movement[switch == 0] = 0
        self._people = self._people + movement

    def change_state(self, lp=14, hrt=0):
        dt = self._round - self._timer
        # 必须先更新时钟再更新状态//Update the timer before updating the status
        # 潜伏期感染患者转确诊//Infected people in latent period become confirmed
        d = np.random.randint(7, lp)
        self._timer[(self._status == 1) & ((dt == d) | (dt > lp))] = self._round
        self._status[(self._status == 1) & ((dt == d) | (dt > lp))] += 1
        # 确证患者转医院隔离//Confirmed cases move into hospital and become isolated
        if self.hospital.count > len(self.isolated[:, 0]):
            empty = self.hospital.count - len(self.isolated[:, 0])
            if empty > len(self._timer[(self._status == 2) & dt >= hrt]):
                self._timer[(self._status == 2) & (dt >= hrt)] = self._round
                self._status[(self._status == 2) & (dt >= hrt)] += 1
            else:
                self._timer[np.where((self._status == 2) & (dt >= hrt))[0][0:empty]] = self._round
                self._status[np.where((self._status == 2) & (dt >= hrt))[0][0:empty]] += 1



    def affect(self, x=0.):
        # self.infect_nearest()
        self.infect_possible(x=x)

    def infect_nearest(self, safe_distance=1.0):
        """感染最接近的健康人//Infect the nearest healthy people"""
        for inf in self.infected:
            dm = (self._people - inf) ** 2
            d = dm.sum(axis=1) ** 0.5
            sorted_index = d.argsort()
            for i in sorted_index:
                if d[i] >= safe_distance:
                    break  # 超出范围，不用管了//Keep static if distance is greater than safe distance
                if self._status[i] > 0:
                    continue
                self._status[i] = 1
                # 记录状态改变的时间//Record timer of status change
                self._timer[i] = self._round
                break  # 只传 1 个//Only infect one nearest healthy person

    def infect_possible(self, x=0., safe_distance=2.0):
        """按概率感染接近的健康人//Infect neighboring healthy people by probability
        x 的取值参考正态分布概率表，x=0 时感染概率是 50%//Infectious rate obeys Gaussian distribution of N(0,1)
        """
        for inf in self.infected:
            dm = (self._people - inf) ** 2
            d = dm.sum(axis=1) ** 0.5
            sorted_index = d.argsort()
            for i in sorted_index:
                if d[i] >= safe_distance:
                    break  # 超出范围，不用管了//Keep static if distance is greater than safe distance
                if self._status[i] > 0:
                    continue
                if np.random.normal() > x:
                    continue
                self._status[i] = 1
                # 记录状态改变的时间//Record timer of status change
                self._timer[i] = self._round

        for inf in self.confirmed:
            dm = (self._people - inf) ** 2
            d = dm.sum(axis=1) ** 0.5
            sorted_index = d.argsort()
            for i in sorted_index:
                if d[i] >= safe_distance:
                    break  # 超出范围，不用管了//Keep static if distance is greater than safe distance
                if self._status[i] > 0:
                    continue
                '''假设确证患者传染率高于潜伏期患者10%左右//
                Assume the infectious rate of confirmed cases is about 10% higher than infected people'''
                if np.random.normal() > x+0.3:
                    continue
                self._status[i] = 1
                # 记录状态改变的时间//Record timer of status change
                self._timer[i] = self._round

    def over(self):
        return len(self.healthy) == 0

    def report(self):
        plt.cla()
        plt.grid(False)
        p1 = plt.scatter(self.healthy[:, 0], self.healthy[:, 1], s=1)
        p2 = plt.scatter(self.infected[:, 0], self.infected[:, 1], s=1, c='orange')
        p3 = plt.scatter(self.confirmed[:, 0], self.confirmed[:, 1], s=1, c='red')
        p4 = plt.scatter(self.hospital.beds[0:len(self.isolated), 0],
                         self.hospital.beds[0:len(self.isolated), 1], s=2, c='green')

        p_hos = plt.Rectangle(xy=(399, -201), width=self.hospital.width+1, height=self.hospital.height+1,
                              color="green", fill=False)
        plt.gca().add_patch(p_hos)

        plt.legend([p1, p2, p3, p4],
                   ['healthy', 'infected', 'confirmed', 'isolated'], loc='upper left', scatterpoints=1)
        t = "Round: %s, Healthy: %s, Infected: %s, Confirmed: %s, Isolated: %s" % \
            (self._round, len(self.healthy), len(self.infected), len(self.confirmed), len(self.isolated))
        plt.text(-200, 400, t, ha='left', wrap=True)
        plt.text(400, -220, 'Hospital')

    def update(self):
        """每一次迭代更新"""
        self.change_state(14, 0)  # 潜伏期及医院收治响应时间//Latent period and hospital response time
        self.affect(5)  # 病毒传染率//Virus infectious rate
        self.move(5, 2)  # 人员移动范围及流动意向//People movement range and mobility intention
        self._round += 1
        self.report()


if __name__ == '__main__':
    np.random.seed(0)
    plt.figure(figsize=(15, 15), dpi=85)  # 城市大小设置//City size
    plt.ion()
    h = Hospital(20, 50)  # 医院大小病床数//Number of beds in hospital
    p = People(5000, 3, h)  # 城市起始人数及起始感染者数目//Initial people and infected cases
    for i in range(200):  # 模拟总天数//Total number of days of simulation
        p.update()
        plt.pause(.05)
    plt.pause(3)
