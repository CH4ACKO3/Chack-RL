import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch import nn
from math import *
from collections import deque
import random
# 动态展示
dyna = False
# 测试(注意初始条件)
test = False

class DuelingNet(nn.Module):
    def __init__(self, n_states, n_actions,hidden_dim=128):
        super(DuelingNet, self).__init__()
        
        # hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU()
        )
        
        #  advantage
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # value
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        x = self.hidden_layer(state)
        advantage = self.advantage_layer(x)
        value     = self.value_layer(x)
        return value + advantage - advantage.mean()

class Env:
    def __init__(self):
        self.v = 1 # 速度
        self.dp = 30/360*pi # 角速度(可用过载)
        self.k = 5 # 奖励随距离衰减的速率
        self.dt = 0.05 # 模拟步长
        self.r, self.sp, self.tp, self.x0, self.y0, self.p0, self.x1, self.y1, self.p1 = np.zeros(9) # 状态初始化

    def restrict(self, a):
        if a>pi:  # 把角度限制在 (-pi,pi]
            a -= 2*pi
        elif a<=-pi:
            a += 2*pi
        return a
    
    def reset(self):
        self.x1, self.y1, self.p1 = [0, 0, 0] # 敌机状态
        self.x0, self.y0 = np.random.random_sample(2)*4-2 # 自机状态
        self.p0 = (np.random.random_sample(1)[0]*2*pi-pi)

        self.r = hypot(self.y1-self.y0, self.x1-self.x0) # 坐标转换
        self.p = atan2(self.y1-self.y0, self.x1-self.x0)
        self.sp = self.restrict(self.p1-self.p) 
        self.tp = self.restrict(self.p0-self.p)

        return [self.r, self.sp, self.tp]

    def step(self, action):
        # state = [delta_r, delta_sp, delt_tp, self_x, self_y, self_p, target_x, target_y, target_p]
        t_action = 1 # 敌机动作

        self.p0 += self.dp*self.dt*(action-1)
        self.p1 += self.dp*self.dt*(t_action-1)

        # 敌机直角机动
        # if self.x1 >= 10: 
        #     self.p1 = pi/2
        # if self.y1 >= 20:
        #     self.p1 = 0
        # if self.x1 >= 30:
        #     self.p1 = -pi/2
        # if self.y1 < 0:
        #     self.p1 = 0

        self.p0 = self.restrict(self.p0)
        self.p1 = self.restrict(self.p1)

        self.x0 += cos(self.p0)*self.v*self.dt
        self.y0 += sin(self.p0)*self.v*self.dt
        self.x1 += cos(self.p1)*self.v*self.dt
        self.y1 += sin(self.p1)*self.v*self.dt

        self.r = hypot(self.y1-self.y0, self.x1-self.x0)
        self.p = atan2(self.y1-self.y0, self.x1-self.x0)
        self.sp = self.restrict(self.p0-self.p)
        self.tp = self.restrict(self.p1-self.p)

        state = [self.r, self.sp, self.tp]
        reward = (1-0.8*abs(self.sp)/pi-0.2*abs(self.tp)/pi)*(1-0.5*exp(-self.r/self.k))
        
        return [state, reward]
    
    def status(self): # 直角坐标状态采样,绘图用
        return [self.x0, self.y0, self.p0, self.x1, self.y1, self.p1]
    
class Agent:
    def __init__(self):
        self.train_eps = 50 # 训练回合数
        self.test_eps = 5 # 测试回合数
        self.max_steps = 2000 # 最大步数
        self.gamma = 0.99 # 衰减系数
        self.learn_rate = 1e-4 # 学习率
        self.epsilon = 0.1 # 探索系数
        self.batch_size = 32 # 批容量
        self.device = 'cpu' # 设备
        self.target_net = DuelingNet(3, 3).to(self.device)
        self.policy_net = DuelingNet(3, 3).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learn_rate, amsgrad=True)
        self.buffer = deque([], maxlen=1000000) # 缓存

    def sample_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        if np.random.random_sample(1)[0] > self.epsilon: # epsilon-greedy
            return self.policy_net(state).max(0)[1].item()
        else:
            return np.random.choice(3)
    
    def predict_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        return self.policy_net(state).max(0)[1].item()
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, nxt_state = list(zip(*random.sample(self.buffer, self.batch_size)))

        # 类型处理
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        nxt_state = torch.tensor(np.array(nxt_state), dtype=torch.float32, device=self.device)
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
        # for param in self.policy_net.parameters():  
            # param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return

    def train(self, env):
        rewards = [] # 回合总回报
        ma_rewards = [] # 平滑回报
        for eps in range(1, 1+self.train_eps): 
            ep_reward = 0 # 重置回合奖励
            state = env.reset() # 重置环境
            for step in range(1, 1+self.max_steps):
                action = self.sample_action(state) # 动作采样
                nxt_state, reward = env.step(action) # 环境交互
                ep_reward += reward # 累加奖励
                self.buffer.append([state, action, reward, nxt_state]) # 存储经验
                self.update()
                state = nxt_state # 更新状态
            self.target_net.load_state_dict(self.policy_net.state_dict()) # 每回合重载
            rewards.append(ep_reward) 
            ma_rewards.append(rewards[-1] if eps == 1 else ma_rewards[-1]*0.8+rewards[-1]*0.2) # 平滑化回报
            print(eps, ep_reward)
        return rewards, ma_rewards

    def test(self, env):
        fig, axe = plt.subplots()
        for i in range(1, 1+self.test_eps):
            state = env.reset()
            ep_reward = 0
            states = []
            for step in range(1, 1+self.max_steps):
                states.append(env.status())
                action = self.predict_action(state)
                nxt_state, reward = env.step(action)
                ep_reward += reward
                state = nxt_state
            x0, y0, p0, x1, y1, p1 = list(zip(*states))
            axe.plot(x0, y0, linewidth=1.0)
            if i == 1:
                axe.plot(x1, y1, linewidth=2.0)
        plt.show()

    def dyna(self, env):
        fig, axe = plt.subplots()
        state = env.reset()
        ep_reward = 0
        states = []
        for step in range(1, 1+self.max_steps):
            states.append(env.status())
            action = self.predict_action(state)
            nxt_state, reward = env.step(action)
            ep_reward += reward
            state = nxt_state
            x0, y0, p0, x1, y1, p1 = list(zip(*states))
            axe.clear() # 动态绘图
            axe.plot(x0, y0)
            axe.plot(x1, y1)
            axe.scatter(x0[-1], y0[-1])
            axe.scatter(x1[-1], y1[-1])
            plt.pause(0.001)
        
agent = Agent()
env = Env()
agent.target_net.load_state_dict(torch.load('chase.pkl')) 
agent.policy_net.load_state_dict(torch.load('chase.pkl'))

if dyna:
    agent.dyna(env)
elif test:
    agent.test(env)
else:
    rewards, ma_rewards = agent.train(env)
    fig, axe = plt.subplots()
    axe.plot(rewards)
    axe.plot(ma_rewards)
    plt.show()
    torch.save(agent.target_net.state_dict(), 'chase.pkl')
    agent.test(env)
    