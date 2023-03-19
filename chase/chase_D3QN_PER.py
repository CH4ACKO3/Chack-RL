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

class SumTree:
    '''SumTree for the per(Prioritized Experience Replay) DQN. 
    This SumTree code is a modified version and the original code is from:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    '''
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data_pointer = 0
        self.n_entries = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype = object)

    def update(self, tree_idx, p):
        '''Update the sampling weight
        '''
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):
        '''Adding new data to the sumTree
        '''
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        # print ("tree_idx=", tree_idx)
        # print ("nonzero = ", np.count_nonzero(self.tree))
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, v):
        '''Sampling the data
        '''
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx] :
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return int(self.tree[0])

class ReplayTree:
    '''ReplayTree for the per(Prioritized Experience Replay) DQN. 
    '''
    def __init__(self, capacity):
        self.capacity = capacity # the capacity for memory replay
        self.tree = SumTree(capacity)
        self.abs_err_upper = 1.

        ## hyper parameter for calculating the importance sampling weight
        self.beta_increment_per_sampling = 0.001
        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 0.01 
        self.abs_err_upper = 1.

    def __len__(self):
        ''' return the num of storage
        '''
        return self.tree.total()

    def push(self, error, sample):
        '''Push the sample into the replay according to the importance sampling weight
        '''
        p = (np.abs(error) + self.epsilon) ** self.alpha
        self.tree.add(p, sample)         


    def sample(self, batch_size):
        '''This is for sampling a batch data and the original code is from:
        https://github.com/rlcode/per/blob/master/prioritized_memory.py
        '''
        pri_segment = self.tree.total() / batch_size

        priorities = []
        batch = []
        idxs = []

        is_weights = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total() 

        for i in range(batch_size):
            a = pri_segment * i
            b = pri_segment * (i+1)

            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
            prob = p / self.tree.total()

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return zip(*batch), idxs, is_weights
    
    def batch_update(self, tree_idx, abs_errors):
        '''Update the importance sampling weight
        '''
        abs_errors += self.epsilon

        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

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
        self.tv = 1
        self.acc = 0.1
        self.t_acc = 0.1
        self.max_v = 1.5
        self.min_v = 0.5
        self.t_max_v = 1.5
        self.t_min_v = 0.5 
        self.dp = 30/180*pi # 角速度(可用过载)
        self.t_dp = 30/180*pi
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
        self.v, self.tv = [1, 1]
        # self.v, self.tv = np.random.random_sample(2)+0.5
        self.x1, self.y1, self.p1 = [0, 0, 0] # 敌机状态
        self.x0, self.y0 = np.random.random_sample(2)*4-2 # 自机状态
        self.p0 = (np.random.random_sample(1)[0]*2*pi-pi)
        # self.p0 = self.restrict(self.p0+pi)
        self.r = hypot(self.y1-self.y0, self.x1-self.x0) # 坐标转换
        self.p = atan2(self.y1-self.y0, self.x1-self.x0)
        self.sp = self.restrict(self.p1-self.p)
        self.tp = self.restrict(self.p0-self.p)

        return [self.r, self.sp, self.tp, self.v, self.tv]

    def step(self, action):
        # state = [delta_r, delta_sp, delt_tp, self_x, self_y, self_p, target_x, target_y, target_p]

        # 自机动作
        if action == 1:
            self.p0 += self.dp*self.dt
        elif action == 2:
            self.p0 -= self.dp*self.dt
        elif action == 3:
            self.v = min(self.max_v, self.v+self.acc)
        elif action == 4:
            self.v = max(self.min_v, self.v-self.acc)

        # 敌机动作
        # t_action = 1 if self.tp>0 else 2
        t_action = 0
        # t_action = np.random.choice(5)
        if t_action == 1:
            self.p1 += self.t_dp*self.dt
        elif t_action == 2:
            self.p1 -= self.t_dp*self.dt
        elif t_action == 3:
            self.tv = min(self.t_max_v, self.tv+self.t_acc)
        elif t_action == 4:
            self.tv = max(self.t_min_v, self.tv-self.t_acc)

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
        self.x1 += cos(self.p1)*self.tv*self.dt
        self.y1 += sin(self.p1)*self.tv*self.dt

        self.r = hypot(self.y1-self.y0, self.x1-self.x0)
        self.p = atan2(self.y1-self.y0, self.x1-self.x0)
        self.sp = self.restrict(self.p0-self.p)
        self.tp = self.restrict(self.p1-self.p)

        state = [self.r, self.sp, self.tp, self.v, self.tv]
        reward = (1-abs(self.sp)/pi-abs(self.tp)/pi)*(0.2+0.8*exp(-self.r/self.k))
        return [state, reward]
    
    def status(self): # 直角坐标状态采样,绘图用
        return [self.x0, self.y0, self.p0, self.x1, self.y1, self.p1]
    
class Agent:
    def __init__(self):
        self.train_eps = 50 # 训练回合数
        self.test_eps = 10 # 测试回合数
        self.max_steps = 500 # 最大步数
        self.gamma = 0.99 # 衰减系数
        self.learn_rate = 1e-4 # 学习率
        self.epsilon = 0.1 # 探索系数
        self.batch_size = 32 # 批容量
        self.device = 'cpu' # 设备
        self.target_net = DuelingNet(5, 5).to(self.device)
        self.policy_net = DuelingNet(5, 5).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learn_rate, amsgrad=True)
        self.buffer = ReplayTree(1000000)

    def sample_action(self, state):
        if np.random.random_sample(1)[0] > self.epsilon: # epsilon-greedy
            with torch.no_grad():
                state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
                return self.policy_net(state).max(0)[1].item()
        else:
            return np.random.choice(5)
    
    def predict_action(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
            return self.policy_net(state).max(0)[1].item()
    
    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        (state, action, reward, nxt_state), idx, is_weight = self.buffer.sample(self.batch_size)

        # 类型处理
        state = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        nxt_state = torch.tensor(np.array(nxt_state), dtype=torch.float32, device=self.device)
        action = torch.tensor(action, device=self.device).unsqueeze_(1)
        reward = torch.tensor(reward, dtype=torch.float32,device=self.device).unsqueeze_(1)
        is_weight = torch.tensor(is_weight, device=self.device)

        q_values = self.policy_net(state).gather(1, action).to(self.device)
        nxt_actions = self.policy_net(nxt_state).max(1)[1].unsqueeze_(1).to(self.device)
        nxt_q_values = self.target_net(nxt_state).gather(1, nxt_actions).to(self.device)
        exp_q_values = reward+self.gamma*nxt_q_values

        loss = torch.mean(torch.pow((q_values-exp_q_values)*is_weight, 2))
        # loss = nn.MSELoss()(q_values, exp_q_values)
        abs_errors = np.sum(np.abs(q_values.cpu().detach().numpy() - exp_q_values.cpu().detach().numpy()), axis=1)
        self.buffer.batch_update(idx, abs_errors) 
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

                policy_val = agent.policy_net(torch.tensor(state, dtype=torch.float32, device = self.device))[action]
                target_val = agent.target_net(torch.tensor(nxt_state, dtype=torch.float32, device = self.device))
                error = abs(policy_val - reward - self.gamma * torch.max(target_val))

                agent.buffer.push(error.cpu().detach().numpy(), (state, action, reward,
                            nxt_state))   # 保存transition
                
                ep_reward += reward # 累加奖励
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
# agent.target_net.load_state_dict(torch.load('chase.pkl')) 
# agent.policy_net.load_state_dict(torch.load('chase.pkl'))

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
    # torch.save(agent.target_net.state_dict(), 'chase.pkl')
    agent.test(env)
    