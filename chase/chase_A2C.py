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

# 动态展示
dyna = True
# 测试(注意初始条件)
test = False

class Agent:
    def __init__(self):
        self.train_eps = 1 #训练回合数
        self.test_eps = 1 #测试回合数
        self.max_steps = 2000 #最大步数
        self.gamma = 0.9 #衰减系数
        self.learn_rate = 2e-3
        self.batch_size = 32 
        self.sample_rounds = 0 #采样次数
        self.device = 'cpu'
        self.actor = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learn_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learn_rate)
        self.buffer = []

    def random_init(self): 
        x1, y1, p1 = [4, 0, 0] #敌机状态
        # p1 = self.restrict(p1+pi)
        x0, y0 = np.random.random_sample(2)*1-0.5 #自机状态
        p0 = (np.random.random_sample(1)[0]*2*pi-pi)/6
        p0 = 0
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
        
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action
        
    def predict_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)

        if state.dim() == 1: #只需要前三个参量
            state = state[[0,1,2]]
        else:
            state = state[:,[0,1,2]]

        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        return action
    
    def restrict(self, a):
        #角度转换
        if a>pi:
            a -= 2*pi
        elif a<=-pi:
            a += 2*pi
        return a
    
    def step(self, state, action):
        # state = [delta_r, delta_sp, delt_tp, self_x, self_y, self_p, target_x, target_y, target_p]
        dt = 0.05 #步长
        v = 1 #速度
        k = 5 #奖励随距离衰减系数
        dp = 30/360*pi
        
        dr, dsp, dtp, x0, y0, p0, x1, y1, p1 = state

        # 敌机动作
        t_action = 1
        # t_action = np.random.randint(3)
        # t_action = self.predict_action(state)

        p0 += dp*dt*(action-1)
        p1 += dp*dt*(t_action-1)

        # 敌机直角机动
        # if x1 >= 10: 
        #     p1 = pi/2
        # if y1 >= 20:
        #     p1 = 0
        # if x1 >= 30:
        #     p1 = -pi/2
        # if y1 < 0:
        #     p1 = 0

        p0 = self.restrict(p0)
        p1 = self.restrict(p1)

        x0 += cos(p0)*v*dt
        y0 += sin(p0)*v*dt
        x1 += cos(p1)*v*dt
        y1 += sin(p1)*v*dt

        r = hypot(y1-y0, x1-x0)
        p = atan2(y1-y0, x1-x0)
        aa = p1-p
        ata = p0-p
        aa = self.restrict(aa)
        ata = self.restrict(ata)

        dr = r
        dsp = ata
        dtp = aa

        aa = abs(aa)
        ata = abs(ata)
        reward = (-0.8*ata/pi-0.2*aa/pi)*(1-0.5*exp(-r/k))

        state = np.array([dr, dsp, dtp, x0, y0, p0, x1, y1, p1])
        # state = [x0, y0, p0, p1]
        return [state, reward]
    
    def update(self):
        state, reward, action, nxt_state = list(zip(*self.buffer))

        state = np.array(state)
        nxt_state = np.array(nxt_state)
        state = state[:,[0,1,2]] #依然只取前三个
        nxt_state = nxt_state[:,[0,1,2]]

        action = torch.tensor(action, device=self.device).unsqueeze_(1)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        nxt_state = torch.tensor(nxt_state, dtype=torch.float32, device=self.device)
        log_prob = torch.log(self.actor(state).gather(1, action))
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze_(1)

        value = self.critic(state)
        target_value = reward+self.gamma*self.critic(nxt_state)
        delta_value = target_value-value
        actor_loss = torch.mean(delta_value.detach()*-log_prob)
        critic_loss =  nn.MSELoss()(value, target_value.detach())
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.buffer = []
        return actor_loss.item(), critic_loss.item()

    def train(self):
        rewards = []
        ave = []
        save = []
        aloss = []
        closs = []
        for eps in range(1, 1+self.train_eps): 
            ep_reward = 0
            state = self.random_init()
            for step in range(self.max_steps):
                action = self.sample_action(state)
                nxt_state, reward = self.step(state, action)
                ep_reward += reward
                self.buffer.append([state, reward, action, nxt_state])
                state = nxt_state
            rewards.append(ep_reward)
            actor_loss, critic_loss = agent.update()
            aloss.append(actor_loss)
            closs.append(critic_loss)

            if eps>2:
                ave.append(ave[-1]*0.9+ep_reward*0.1)
                save.append(ave[-1]*0.9+ave[-1]*0.1)
            elif eps>1:
                ave.append(ave[-1]*0.9+ep_reward*0.1) #平滑奖励曲线
                save.append(ave[-1])
            else:
                ave.append(ep_reward)
                save.append(ep_reward)
            print(eps, ep_reward)
        return rewards, ave, save, aloss, closs

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
agent.actor.load_state_dict(torch.load('chase_actor.pkl'))
agent.critic.load_state_dict(torch.load('chase_critic.pkl'))
if dyna:
    agent.dyna()
elif test:
    agent.test()
else:
    rewards, ave, save, aloss, closs = agent.train()
    fig, axe = plt.subplots(2)
    axe[0].plot(rewards)
    # axe.plot(ave)
    axe[0].plot(save)
    axe[1].plot(aloss, label='actor_loss')
    axe[1].plot(closs, label='critic_loss')
    axe[1].legend()
    plt.show()
    torch.save(agent.actor.state_dict(), 'chase_actor.pkl')
    torch.save(agent.critic.state_dict(), 'chase_critic.pkl')
    agent.test()
    
