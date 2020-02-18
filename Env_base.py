import numpy as np
import random
import copy
import gym
from gym import spaces

MISS_PENALTY = 10

class basicEnv(gym.Env):

    metadata = {'render.modes':['LearnToRoute']}

    def __init__(self, num_v, num_e, graph, capacity, miss_penalty =10,k=10, gamma=100, ep_length=100, ttl=64,sparsity=0.3,dm_seq_len=20,eps=1e-10):
        
        super(basicEnv, self).__init__()
        
        # Define parameters
        self.k = k
        self.num_v = num_v
        self.num_e = num_e
        self.current_step = 0
        self.ep_length = ep_length
        self.ttl = ttl
        self.graph = graph
        self.sparsity = sparsity
        self.dm_seq_len = dm_seq_len
        self.eps = eps
        self.gamma = gamma
        self.capacity = capacity
        self.miss_penalty = miss_penalty
        self.flow_num = int(np.floor(self.sparsity*self.num_v*(self.num_v-1)))
        # Define action space
        self.action_space = spaces.Box(
                low=-1.0,high=1.0,shape=(num_v*num_e,))

        # Define observation space
        self.observation_space = spaces.Box(
                low=0.0,high=np.float32(1e6),shape=(num_v,num_v,k))
        
        self.state = np.zeros(shape=(num_v,num_v,k),dtype=np.float32) # demand matrices
        # self.topo = np.zeros(num_v,num_v)
        self.reset()

    
    def step(self, action):
        
        assert self.action_space.contains(action), "action type {} is invalid.".format(type(action))
        state = self.state
        action_mat = action.reshape((self.num_v,self.num_e))
        reward = self.get_reward(action_mat)

        # get next state
        state = np.roll(state, -1, axis=-1)
        new_dm = self.predict_next_dm()
        state[:,:,-1] = new_dm
        self.state = state

        self.current_step += 1
        done = self.current_step >= self.ep_length

        return self.state, reward, done, {}


    def reset(self):
        self.current_step = 0
        self.generate_fixed_dm()
        # get state
        state = self.state
        state = np.roll(state, -1, axis=-1)
        new_dm = self.predict_next_dm()
        state[:,:,-1] = new_dm
        self.state = state
        return self.state

    def render(self, mode='LearnToRoute'):
        pass
    
    def predict_next_dm(self):
        # cyclic bimodal DM
        return self.fixed_dm[:,:,self.current_step % self.dm_seq_len]


    def generate_fixed_dm(self):

        p = self.sparsity
        q = self.dm_seq_len
        assert (p>=0 and p<=1), "Wrong value for sparsity p"
        fixed_dm = np.zeros((self.num_v,self.num_v,q))
        num_flow = int(np.floor(p*self.num_v*(self.num_v-1)))
        for i in range(q):
            for _ in range(num_flow):
                src = random.randint(0, self.num_v - 1)
                dst = random.randint(0, self.num_v - 1)
                while dst == src or fixed_dm[src,dst,i]!=0:
                    src = random.randint(0, self.num_v - 1)
                    dst = random.randint(0, self.num_v - 1)
                flow_demand = random.uniform(5,65)
                fixed_dm[src,dst,i] = flow_demand

        self.fixed_dm = fixed_dm
    
    
    def get_reward(self, action):
        
        split_weight = action
        demand = self.state[:,:,-1]
        self.traffic = {}
        self.no_where_to_go = 0

        total_arrived = 0
        utilization = np.zeros(self.num_e)

        #punish = 0
        split_mat = np.zeros((self.num_v,self.num_v,self.num_v))
        for dst in range(self.num_v):
            for src in range(self.num_v):
                if src==dst:
                    split_mat[dst,src,dst] = 1
                else:    
                    total_weight = 0
                    for neighbor, link in self.graph[src].items():
                        split_portion = np.exp(-self.gamma*split_weight[dst,link])
                        total_weight += split_portion
                        split_mat[dst,src,neighbor] = split_portion
                    split_mat[dst,src,:] = split_mat[dst,src,:]/total_weight
        
        for dst in range(self.num_v):
            split_mat_dst = split_mat[dst,:,:]
            for src in range(self.num_v):
                demand_sd = demand[src,dst]
                if demand_sd != 0:
                    t = 0
                    util_flow = np.zeros(self.num_e)
                    F_prev = np.zeros(self.num_v)
                    f_prev = np.zeros(self.num_v)
                    F_prev[src] = 1.0 
                    f_prev[src] = 1.0
                    while t < self.ttl:
                        F_t, util_plus = self.spread_traffic_step(src,dst,split_mat_dst,F_prev)
                        f_t = self.spread_traffic_flowt0(src,dst,split_mat_dst,f_prev)
                        util_flow += util_plus
                        F_prev = F_t
                        f_prev = f_t
                        t += 1
                    arrived = f_t[dst]

                    total_arrived += arrived

                    utilization += util_flow/(self.ttl+1)

        mlu = np.max(utilization)
        util_cap_dif = np.clip(utilization-self.capacity,0,mlu)

        reward =  - (mlu+7*np.max(util_cap_dif))/np.sum(demand) + self.miss_penalty * (total_arrived - self.flow_num)/self.flow_num
        print("mlu {}".format(mlu))
        print("arrived {0} utilexd {1}".format(total_arrived,np.max(util_cap_dif)))
        print("[Epoch {0}][Reward {1}]".format(self.current_step,reward))
        print("============================================================\n")
        return reward

    def spread_traffic_step(self,src,dst,split_mat,F_t_minus_1):
        util_plus = np.zeros(self.num_e)
        for node in range(self.num_v):
            for neighbor,link in self.graph[node].items():
                util_plus[link] += F_t_minus_1[node]*split_mat[node,neighbor]
        F_t = np.matmul(split_mat.T,F_t_minus_1)
        F_t[src] += 1
        return F_t, util_plus

    def spread_traffic_flowt0(self,src,dst,split_mat,f_t_minus_1):
        f_t = np.matmul(split_mat.T,f_t_minus_1)
        return f_t