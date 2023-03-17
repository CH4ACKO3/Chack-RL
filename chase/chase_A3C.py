import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch import nn
from math import *
from multiprocessing import Pipe, Process
# 动态展示
dyna = False
# 测试(注意初始条件)
test = False

def worker(worker_remote, centre_remote, env):
    centre_remote.close()
    while True:
        cmd, data = worker_remote.recv()
        if cmd == 'step':
            nxt_state, reward = env.step(data)
            worker_remote.send([nxt_state, reward])
        elif cmd == 'reset':
            state = env.reset()
            worker_remote.send(state)

class AsyncVecEnv:
    def __init__(self, envs):
        self.waiting = True
        self.n = len(envs)
        self.centre_remotes, self.worker_remotes = zip(*[Pipe() for _ in range(self.n)])
        self.ps = [Process(target=worker, args=(worker_remote, centre_remote, env))
                    for (worker_remote, centre_remote, env) in zip(self.worker_remotes, self.centre_remotes, envs)]
        for p in self.ps:
            p.start()
        for worker_remote in self.worker_remotes:
            worker_remote.close()
    
    def step_async(self, actions):
        for centre_remote, action in zip(self.centre_remotes, actions):
            centre_remote.send(('step', action))
        self.waiting = True
    
    def step_wait(self):
        results = [centre_remote.recv() for centre_remote in self.centre_remotes]
        self.waiting = False
        nxt_state, rewards = zip(*results)
        return [nxt_state, rewards]
        
    def reset(self):
        for centre_remotes in self.centre_remotes:
            centre_remotes.send(('reset', None))
        return [centre_remote.recv() for centre_remote in self.centre_remotes]
    
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
    
    def __len__(self):
        return self.n
    
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
        self.x1, self.y1, self.p1 = [4, 0, 0] # 敌机状态
        self.x0, self.y0 = np.random.random_sample(2)*1-0.5 # 自机状态
        self.p0 = (np.random.random_sample(1)[0]*2*pi-pi)/6

        self.r = hypot(self.y1-self.y0, self.x1-self.x0) # 坐标转换
        self.p = atan2(self.y1-self.y0, self.x1-self.x0)
        self.sp = self.restrict(self.p1-self.p) 
        self.tp = self.restrict(self.p0-self.p)

        return [self.r, self.sp, self.tp]

    def step(self, action):
        # state = [delta_r, delta_sp, delt_tp, self_x, self_y, self_p, target_x, target_y, target_p]
        t_action = 1 # 敌机动作
        action = action.item()
        self.p0 += self.dp*self.dt*(action-1)
        self.p1 += self.dp*self.dt*(t_action-1)

        # 敌机直角机动
        # if x1 >= 10: 
        #     p1 = pi/2
        # if y1 >= 20:
        #     p1 = 0
        # if x1 >= 30:
        #     p1 = -pi/2
        # if y1 < 0:
        #     p1 = 0

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
        reward = (-abs(self.sp)/pi-abs(self.tp)/pi)*(1-0.5*exp(-self.r/self.k))
        
        return [state, reward]
    
    def status(self): # 直角坐标状态采样,绘图用
        return [self.x0, self.y0, self.p0, self.x1, self.y1, self.p1]
    
class Agent:
    def __init__(self):
        self.train_eps = 200 # 训练回合数
        self.test_eps = 10 # 测试回合数
        self.max_steps = 300 # 最大步数
        self.gamma = 0.9 # 衰减系数
        self.learn_rate = 2e-3 # 学习率
        self.n_envs = 6 # 并行进程数
        self.device = 'cpu' # 设备
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
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learn_rate) # 优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learn_rate)
        self.buffer = [] # 缓存

    def sample_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)

        probs = self.actor(state) # 动作概率分布
        dist = torch.distributions.Categorical(probs) # 采样
        action = dist.sample()

        return action
        
    def update(self):
        state, reward, action, nxt_state = list(zip(*self.buffer)) # 从缓存中提取经验
        action = torch.tensor(action, device=self.device).unsqueeze_(1) # 数据处理
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        nxt_state = torch.tensor(np.array(nxt_state), dtype=torch.float32, device=self.device)
        log_prob = torch.log(self.actor(state).gather(1, action))
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze_(1)

        value = self.critic(state) # 继续处理
        target_value = reward+self.gamma*self.critic(nxt_state)
        delta_value = target_value-value

        actor_loss = torch.mean(delta_value.detach()*-log_prob) # 计算损失函数
        critic_loss =  nn.MSELoss()(value, target_value.detach())

        self.actor_optimizer.zero_grad()  # 梯度清零
        self.critic_optimizer.zero_grad()
        actor_loss.backward() # 反向传播
        critic_loss.backward()
        self.actor_optimizer.step() # 梯度更新
        self.critic_optimizer.step()

        self.buffer = [] # 缓存清零
        return actor_loss.item(), critic_loss.item()

    def train(self, envs):
        returns = [] # 回合总回报
        ma_returns = [] # 平滑回报
        loss = [] # 回合损失
        for eps in range(1, 1+self.train_eps): 
            ep_reward = 0 # 重置回合奖励
            states = envs.reset() # 重置环境
            for step in range(1, 1+self.max_steps):
                actions = self.sample_action(states) # 动作采样
                nxt_states, rewards = envs.step(actions) # 环境交互
                ep_reward += np.array(rewards).mean() # 累加奖励
                for i in range(self.n_envs):
                    self.buffer.append([states[i], rewards[i], actions[i], nxt_states[i]]) # 存储经验
                states = nxt_states # 更新状态
            returns.append(ep_reward) 
            actor_loss, critic_loss = agent.update() # 更新神经网络
            loss.append([actor_loss, critic_loss]) 
            ma_returns.append(returns[-1] if eps == 1 else ma_returns[-1]*0.9+returns[-1]*0.1) # 平滑化回报
            print(eps, ep_reward)
        return returns, ma_returns, loss

    def test(self):
        env = Env()
        fig, axe = plt.subplots()
        for i in range(1, 1+self.test_eps):
            state = env.reset()
            ep_reward = 0
            states = []
            for step in range(1, 1+self.max_steps):
                states.append(env.status())
                action = self.sample_action(state)
                nxt_state, reward = env.step(action)
                ep_reward += reward
                state = nxt_state
            x0, y0, p0, x1, y1, p1 = list(zip(*states))
            axe.plot(x0, y0, linewidth=1.0)
            if i == 1:
                axe.plot(x1, y1, linewidth=2.0)
        plt.show()

    def dyna(self):
        env = Env()
        fig, axe = plt.subplots()
        state = env.reset()
        ep_reward = 0
        states = []
        for step in range(1, 1+self.max_steps):
            states.append(env.status())
            action = self.sample_action(state)
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

if __name__ == '__main__':
    agent = Agent()
    env = Env()
    agent.actor.load_state_dict(torch.load('chase_actor.pkl')) # 读取模型
    agent.critic.load_state_dict(torch.load('chase_critic.pkl'))

    envs = [Env() for i in range(agent.n_envs)]
    envs = AsyncVecEnv(envs)

    if dyna:
        agent.dyna()
    elif test:
        agent.test()
    else:
        rewards, ma_rewards, loss = agent.train(envs)

        fig, axe = plt.subplots(2)
        axe[0].plot(rewards)
        axe[0].plot(ma_rewards)
        aloss, closs = list(zip(*loss))
        axe[1].plot(aloss, label='actor_loss')
        axe[1].plot(closs, label='critic_loss')
        axe[1].legend()
        plt.show()

        torch.save(agent.actor.state_dict(), 'chase_actor.pkl') # 保存模型
        torch.save(agent.critic.state_dict(), 'chase_critic.pkl')
        agent.test()
        
