import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time, math
import torch
from torch import optim
from torch import nn
from collections import deque
from math import *
import random

# 动态展示,参考最下方代码
dyna = False
# 测试(注意初始条件)
test = False

class Agent:
    def __init__(self):
        self.train_eps = 1 #训练回合数
        self.test_eps = 10 #测试回合数
        self.max_steps = 500 #最大步数
        self.gamma = 0.99 #衰减系数
        self.epsilon_start = 0.9 
        self.epsilon_end = 0.05 
        self.epsilon_decay = 0.00001 
        self.epsilon = self.epsilon_start 
        self.learn_rate = 1e-4 
        self.batch_size = 32 
        self.sample_rounds = 0 #采样次数
        self.device = 'cpu'
        self.target_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        ).to(self.device)
        self.policy_net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = self.learn_rate)
        self.buffer = deque([], maxlen=1000000)
        self.int = 0

    def random_init(self): 
        x1, y1, p1 = [4, 0, 0] # 敌机状态
        # p1 = 0/180*pi

        x0, y0 = np.random.random_sample(2)*4-2 # 自机状态,可以改着玩,或者设计课程学习
        p0 = (np.random.random_sample(1)[0]*2*pi-pi)
        # p0 = 0
        # p0 = atan2(y0, x0)
        # x0, y0, p0 = [0, -3, pi/4]
        # p0 = self.restrict(p0+pi)

        r = hypot(y1-y0, x1-x0) #坐标转换
        p = atan2(y1-y0, x1-x0)
        aa = self.restrict(p1-p)
        ata = self.restrict(p0-p)

        return np.array([r, ata, aa, x0, y0, p0, x1, y1, p1])
        
    def sample_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)

        if state.dim() == 1: #只需要前三个参量
            state = state[[0,1,2]]
        else:
            state = state[:,[0,1,2]]

        self.epsilon = self.epsilon_end+(self.epsilon_start-self.epsilon_end)*math.exp(-self.epsilon_decay*self.sample_rounds)
        self.sample_rounds += 1

        if random.uniform(0, 1) > self.epsilon: # epsilon-greedy
            return self.policy_net(state).max(0)[1].item()
        else:
            return np.random.choice(3)
        
    def predict_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)

        if state.dim() == 1:
            state = state[[0,1,2]]
        else:
            state = state[:,[0,1,2]]

        return self.policy_net(state).max(0)[1].item()
    
    def restrict(self, a):
        # 将角度限制在 [-pi, pi) 之间
        if a>pi:
            a -= 2*pi
        elif a<=-pi:
            a += 2*pi
        return a
    
    def step(self, state, action):
        # state = [delta_r, delta_sp, delt_tp, self_x, self_y, self_p, target_x, target_y, target_p]
        dt = 0.02 # 步长
        v = 1 # 速度
        k = 5 # 奖励随距离衰减系数
        dp = 60/360*pi # 每秒角度变化率(可用过载)
        
        dr, dsp, dtp, x0, y0, p0, x1, y1, p1 = state

        # 敌机动作
        t_action = 1
        # t_action = np.random.randint(3)
        # t_action = self.predict_action(state)

        # 角度变化
        p0 += dp*dt*(action-1)
        p1 += dp*dt*(t_action-1)

        # 敌机直角机动
        if x1 >= 10: 
            p1 = pi/2
        if y1 >= 10:
            p1 = 0
        if x1 >= 50:
            p1 = -pi/2
        if y1 < 0:
            p1 = 0

        p0 = self.restrict(p0)
        p1 = self.restrict(p1)

        # 位置变化
        x0 += cos(p0)*v*dt
        y0 += sin(p0)*v*dt
        x1 += cos(p1)*v*dt
        y1 += sin(p1)*v*dt

        r = hypot(y1-y0, x1-x0) # 距离
        p = atan2(y1-y0, x1-x0) # 视线角
        aa = p1-p # 敌机朝向与视线的夹角
        ata = p0-p # 自机朝向与视线的夹角
        aa = self.restrict(aa)
        ata = self.restrict(ata)

        dr = r
        dsp = ata
        dtp = aa

        aa = abs(aa) # 求奖励的时候用绝对值
        ata = abs(ata)
        reward = (2-1.6*ata/pi-0.4*aa/pi)*(0.2+0.8*exp(-r/k)) # 目标是跟随到敌机的正后方
        state = np.array([dr, dsp, dtp, x0, y0, p0, x1, y1, p1])
        # state = [x0, y0, p0, p1]
        return [state, reward]
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, nxt_state = list(zip(*random.sample(self.buffer, self.batch_size)))

        state = np.array(state)
        nxt_state = np.array(nxt_state)
        state = state[:,[0,1,2]] # 依然只取前三个
        nxt_state = nxt_state[:,[0,1,2]]

        # 类型处理
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        nxt_state = torch.tensor(nxt_state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, device=self.device).unsqueeze_(1)
        reward = torch.tensor(reward, dtype=torch.float32,device=self.device).unsqueeze_(1)

        q_values = self.policy_net(state).gather(1, action).to(self.device)
        nxt_actions = self.policy_net(nxt_state).max(1)[1].unsqueeze_(1).to(self.device)
        nxt_q_values = self.target_net(nxt_state).gather(1, nxt_actions).to(self.device)
        exp_q_values = reward+self.gamma*nxt_q_values

        loss = nn.MSELoss()(q_values, exp_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # 避免梯度过大
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self):
        rewards = []
        ave = []
        for eps in range(1, 1+self.train_eps): 
            ep_reward = 0
            state = self.random_init()
            for step in range(self.max_steps):
                action = self.sample_action(state)
                nxt_state, reward = self.step(state, action)
                ep_reward += reward
                self.buffer.append([state, action, reward, nxt_state])
                self.update() # 每步更新
                state = nxt_state
            self.target_net.load_state_dict(self.policy_net.state_dict()) # 每回合重载
            rewards.append(ep_reward)
            if eps>1:
                ave.append(ave[-1]*0.9+ep_reward*0.1) # 平滑奖励曲线
            else:
                ave.append(ep_reward)
            print(eps, ep_reward)
        return rewards, ave

    def test(self):
        fig, axe = plt.subplots()
        for i in range(self.test_eps):
            state = self.random_init()
            ep_reward = 0
            states = []
            for step in range(self.max_steps):
                states.append(state)
                action = self.predict_action(state)
                # action = 1
                nxt_state, reward = self.step(state, action)
                ep_reward += reward
                state = nxt_state
            dr, dsp, dtp, x0, y0, p0, x1, y1, p1 = list(zip(*states))
            axe.plot(x0, y0, linewidth=1.0)
            if i == 0 :
                axe.plot(x1, y1, linewidth=2.0)
        # axe.set_xlim(-1, 30)
        # axe.set_ylim(-10, 10)
        plt.show()

    def dyna(self):
        fig, axe = plt.subplots()
        state = self.random_init()
        ep_reward = 0
        states = []
        for step in range(self.max_steps):
            states.append(state[[3,4,6,7]])
            action = self.predict_action(state)
            # action = 1
            nxt_state, reward = self.step(state, action)
            ep_reward += reward
            state = nxt_state
            x0, y0, x1, y1 = list(zip(*states))
            axe.clear()
            axe.plot(x0, y0)
            axe.plot(x1, y1)
            axe.scatter(x0[-1], y0[-1])
            axe.scatter(x1[-1], y1[-1])
            # plt.show()
            plt.pause(0.001)
        # axe.set_xlim(-1, 30)
        # axe.set_ylim(-10, 10)
        
agent = Agent()
# 加载模型,想重新训练就注释掉
# agent.target_net.load_state_dict(torch.load('chase.pkl')) 
# agent.policy_net.load_state_dict(torch.load('chase.pkl'))
if dyna:
    agent.dyna()
elif test:
    agent.test()
else:
    rewards, ave = agent.train()
    fig, axe = plt.subplots()
    axe.plot(rewards)
    axe.plot(ave)
    plt.show()
    # 保存模型,可以注释掉
    # torch.save(agent.target_net.state_dict(), 'chase.pkl')
    agent.test()
    
