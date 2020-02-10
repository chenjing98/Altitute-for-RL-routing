import numpy as np
import random
import copy
import gym
from gym import spaces
#from gurobipy import *

MISS_PENALTY = 10

class ALTEnv(gym.Env):

    metadata = {'render.modes':['LearnToRoute']}

    def __init__(self, num_v, num_e, graph, capacity, miss_penalty = 10,k=10, gamma=100, ep_length=100, max_hop=300,sparsity=0.3,dm_seq_len=20,eps=1e-10):
        
        super(ALTEnv, self).__init__()
        
        # Define parameters
        self.k = k
        self.num_v = num_v
        self.num_e = num_e
        self.current_step = 0
        self.ep_length = ep_length
        self.max_hop = max_hop
        self.graph = graph
        self.sparsity = sparsity
        self.dm_seq_len = dm_seq_len
        self.eps = eps
        self.gamma = gamma
        #self.corr_range = 0.4
        self.corr_range = num_v/2 - 0.6
        self.num_epoch = 0
        self.miss_penalty = miss_penalty
        self.flow_num = int(np.floor(self.sparsity*self.num_v*(self.num_v-1)))
        self.capacity = capacity

        # Define action space
        self.action_space = spaces.Box(
                low=-1.0,high=1.0,shape=(num_v*(num_v-1+num_e),))
        #print(self.action_space.low,self.action_space.high)
        # Define observation space
        self.observation_space = spaces.Box(
                low=0.0,high=np.float32(1e6),shape=(num_v,num_v,2))
        
        self.state = np.zeros(shape=(num_v,num_v,2),dtype=np.float32) # demand matrices
        # self.topo = np.zeros(num_v,num_v)
        self.reset()

    
    def step(self, action):
        
        assert self.action_space.contains(action), "action type {} is invalid.".format(type(action))
        action_mat = action.reshape((self.num_v,self.num_v-1+self.num_e))
        reward = self.get_reward(action_mat)

        state = np.zeros((self.num_v,self.num_v,2))
        new_dm = self.generate_dm()
        state[:,:,0] = new_dm
        new_capacity = self.generate_capacity()
        state[:,:,1] = new_capacity
        self.state = state

        self.current_step += 1
        done = self.current_step >= self.ep_length
        if done:
            self.num_epoch += 1

        return self.state, reward, done, {}


    def reset(self):
        self.current_step = 0
        #self.generate_fixed_dm()
        # get state
        state = np.zeros((self.num_v,self.num_v,2))
        new_dm = self.generate_dm()
        state[:,:,0] = new_dm
        new_capacity = self.generate_capacity()
        state[:,:,1] = new_capacity
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

    def generate_dm(self):

        p = self.sparsity
        assert (p>=0 and p<=1), "Wrong value for sparsity p"
        dm = np.zeros((self.num_v,self.num_v))
        num_flow = int(np.floor(p*self.num_v*(self.num_v-1)))
        for _ in range(num_flow):
            src = random.randint(0, self.num_v - 1)
            dst = random.randint(0, self.num_v - 1)
            while dst == src or dm[src,dst]!=0:
                src = random.randint(0, self.num_v - 1)
                dst = random.randint(0, self.num_v - 1)
            #flow_demand = random.uniform(5,65)
            lam = np.mean(self.capacity)
            flow_demand = random.uniform(0.1*lam,0.5*lam)
            dm[src,dst] = flow_demand

        return dm
    
    def generate_capacity(self):
        cap = self.capacity
        cap_mat = capacity_vec2mat(cap,self.num_v,self.num_e,self.graph)
        return cap_mat

    
    def get_reward(self, action):
        
        self.altitude = naive_altitude(self.num_v,self.graph)
        modf = np.zeros((self.num_v,self.num_v))
        for i in range(self.num_v-1):
            modf[i+1,:(i+1)] = action[i+1,:(i+1)]
            modf[i,(i+1):] = action[i,i:(self.num_v-1)]
        modf_norm = np.zeros_like(modf)

        modf_norm = (self.num_v/2-1)*modf*modf*modf
        corr_range_mat = self.corr_range * np.ones_like(modf_norm)
        upper = np.clip(modf_norm-corr_range_mat,0,self.num_v)
        lower = np.clip(-corr_range_mat-modf_norm,0,self.num_v)
        exceeded_up = upper>lower
        exceeded_lo = upper<=lower
        exceeded = exceeded_up * upper + exceeded_lo * lower
        exceeded_punish = np.mean(exceeded)
        modf_norm = np.clip(modf_norm,-self.corr_range,self.corr_range)
        self.altitude = self.altitude + modf_norm
        #print(self.altitude)
        #self.altitude = naive_altitude(self.num_v,self.graph)
        #modf = np.zeros_like(self.altitude)
        #for i in range(self.num_v):
        #    modf[:,i] = i/self.num_v * np.ones_like(modf[:,i])
        #self.altitude = self.altitude + modf
        split_weight = action[:,(self.num_v-1):]
        demand = self.state[:,:,0]
        capacity = self.state[:,:,1]
        capacity_vec = capacity_mat2vec(capacity,self.num_v,self.num_e,self.graph)
        self.traffic = {}
        self.no_where_to_go = 0
        #print("====================== STEP {} =======================".format(self.current_step))
        #print("------------------------ Agent action ------------------------")
        #miss_count = 0
        total_arrived = 0
        self.exceed_max_search = False
        # softmin, dst-based routing
        self.hop_count = 0
        utilization = np.zeros(self.num_e)
        for src in range(self.num_v):
            for dst in range(self.num_v):
                if(demand[src,dst]==0):
                    continue
                self.exceed_max_search = False
                self.hop_count = 0
                if (src,dst) in self.traffic:
                    util_split, arrived = self.traffic[(src,dst)]
                else:
                    util_split, arrived = self.spread_traffic(src,dst,split_weight[dst],True)
                utilization += demand[src,dst]*util_split
                total_arrived += arrived
                #if self.exceed_max_search:
                #    miss_count += 1
                if self.exceed_max_search:
                    print("[flow ({0},{1})] arrived:{2}".format(src,dst,arrived))
        mlu = np.max(utilization)
        util_cap_dif = np.clip(utilization-capacity_vec,0,mlu)
        util_pnl = mlu + 7 * util_cap_dif
        #print("-------------------------------------------------------------")
        #print(self.traffic)
        #print("-------------------------------------------------------------")
        self.traffic = {}

        # calculate optimal mlu 
        #print("------------------------ Baseline ------------------------")
        ## with Gurobi (as normalization)
        #try:
        #    opt_model = Model('least-mlu')    
        #except GurobiError:
        #    print('Error reported')
        self.altitude = naive_altitude(self.num_v,self.graph)
        total_arrived_base = 0.0
        self.exceed_max_search = False
        allone_weight = np.ones_like(split_weight)
        self.hop_count = 0
        utilization_base = np.zeros(self.num_e)
        for src in range(self.num_v):
            for dst in range(self.num_v):
                if(demand[src,dst]==0):
                    continue
                self.exceed_max_search = False
                self.hop_count = 0
                if (src,dst) in self.traffic:
                    util_split_base, arrived_base = self.traffic[(src,dst)]
                else:  
                    util_split_base, arrived_base = self.spread_traffic(src,dst,allone_weight[dst],False)
                utilization_base += demand[src,dst]*util_split_base
                total_arrived_base += arrived_base
                #if arrived_base < 0.000001 or self.exceed_max_search:
                #    print("[flow ({0},{1})] arrived:{2}".format(src,dst,arrived_base))
        baseline_mlu = np.max(utilization_base)
        util_cap_dif_base = np.clip(utilization_base-capacity_vec,0,baseline_mlu)
        util_pnl_base = utilization_base + 7 * util_cap_dif_base
        #print("------------------------------------------------------------")
        #reward = - mlu/baseline_mlu + MISS_PENALTY * (total_arrived - 54)/54 - exceeded_punish #- self.no_where_to_go
        reward = - (np.max(util_pnl)/np.max(util_pnl_base)-1) + self.miss_penalty * (total_arrived - self.flow_num)/self.flow_num #- exceeded_punish

        #reward = - 5 * (mlu/baseline_mlu - 1)
        """
        if(total_arrived==54 and self.corr_range<(self.num_v/2-0.6)):
            self.corr_range += 0.5
        if(total_arrived<53 and self.corr_range>0.4):
            self.corr_range -= 0.5
        """
        #if self.num_epoch != 0 and self.num_epoch % 2000000 == 0 and self.corr_range<(self.num_v/2-0.6):
        #    self.corr_range += 0.5
        print("num_epoch {}".format(self.num_epoch))
        print("correction_range {0} exceede_punish {1}".format(self.corr_range, exceeded_punish))
        print("mlu {0}  baseline_mlu {1} arrived {2}".format(mlu,baseline_mlu,total_arrived))
        print("exceed: NN {0} baseline {1}".format(np.max(util_cap_dif), np.max(util_cap_dif_base)))
        #print("total_arrived {0}  total_arrived_base {1}".format(total_arrived,total_arrived_base))
        print("[Epoch {0}][Reward {1}]".format(self.current_step,reward))
        print("============================================================\n")
        return reward

    def spread_traffic(self, src, dst, link_weight,flag):

        if src == dst:
            #print("src:{0} dst:{1} split:{2}".format(src,dst,np.zeros(self.num_e)))
            self.traffic[(src,dst)]=(np.zeros(self.num_e),1.0)
            return np.zeros(self.num_e), 1.0
        elif self.exceed_max_search:
            #self.traffic[(src,dst)]=(np.zeros(self.num_e),0.0)
            return np.zeros(self.num_e), 0.0
        else:
            self.hop_count += 1
            total_weight = 0.0
            util = np.zeros(self.num_e)
            arrived_flow = 0.0
            #prev1 = copy.deepcopy(prev)
            #prev1.append(src)
            for neighbor, link in self.graph[src].items():
                if(self.hop_count>=self.max_hop):
                    self.exceed_max_search = True
                if self.altitude[dst,neighbor] <= self.altitude[dst,src]:
                    continue
                split_portion = np.exp(-self.gamma*link_weight[link])
                total_weight += split_portion
                if (neighbor,dst) in self.traffic:
                    util_plus, arrived_plus = self.traffic[(neighbor,dst)]
                else:
                    util_plus, arrived_plus = self.spread_traffic(neighbor, dst, link_weight,flag)
                util[link] += split_portion
                util += split_portion * util_plus
                arrived_flow += split_portion * arrived_plus
            if total_weight == 0:
                # self.exceed_max_search = True
                self.traffic[(src,dst)]=(np.zeros(self.num_e),0.0)
                if flag:
                    self.no_where_to_go += 1
                return np.zeros(self.num_e), 0.0
            util = util/total_weight
            arrived_flow = arrived_flow/total_weight
            #print("src:{0} dst:{1} split:{2}".format(src,dst,util))
            if (src,dst) not in self.traffic:
                self.traffic[(src,dst)]=(util,arrived_flow)
            #if (util-1.0).any() > 0:
            #    print([src,dst])

            return util, arrived_flow

def naive_altitude(num_v, graph):
    num_hop = num_v*np.ones((num_v,num_v))
    for i in range(num_v):
        num_hop[i,i] = 0
    while True:
        flag = True
        for dst in range(num_v):
            for src in range(num_v):
                for neighbor in graph[src]:
                    if (num_hop[dst][neighbor] + 1) < num_hop[dst][src]:
                        num_hop[dst][src] = num_hop[dst][neighbor] + 1
                        flag = False
                        #print("dst:{0} src:{1} neighbor:{2} hop:{3}".format(dst,src,neighbor,num_hop[dst][src]))
        if flag:
            break
    for i in range(num_v):
        num_hop[i,i] = -1-num_v
    alt = num_v *np.ones_like(num_hop)-num_hop
    return alt

def capacity_vec2mat(vec,num_v,num_e,topo):
    capacity_mat = np.zeros((num_v,num_v))
    for v in range(num_v):
        for neighbor, link in topo[v].items():
            capacity_mat[v,neighbor] = vec[link]
    return capacity_mat

def capacity_mat2vec(mat,num_v,num_e,topo):
    capacity_vec = np.zeros(num_e)
    for v in range(num_v):
        for neighbor, link in topo[v].items():
            capacity_vec[link] = mat[v, neighbor]
    return capacity_vec