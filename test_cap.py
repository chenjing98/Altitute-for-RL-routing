import numpy as np
import random

import Env_cap as env

from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

MODEL_NAME = "ppo2_alt_geant"
MODEL_NAME_BASE = "ppo2_base_geant2"
S_LEN = 1
EP_LENGTH = 1
MAX_HOP = 500

"""
# NSFNet topology
NUM_VERTICE = 14
NUM_EDGE = 21
TOPOLOGY = [{1:1,2:2,3:3},{0:1,2:4,7:7},{0:2,1:4,5:5},{0:3,4:6,8:9},{3:6,5:8,6:0},{2:5,4:8,12:18,13:19},\
         {4:0,7:10},{1:7,6:10,10:11},{3:9,9:13,11:14},{8:13,10:20,12:16},{7:11,9:20,11:12,13:17},\
         {8:14,10:12,12:15},{5:18,9:16,11:15},{5:19,10:17}] 
CAPACITY = [110,110,110,110,110,110,140,110,140,110,110,140,110,110,110,110,110,110,110,110,110] # kbps
"""

# GEANT2 topology
NUM_VERTICE = 24
NUM_EDGE = 37
TOPOLOGY = [{1:0,2:18},{0:0,3:19,6:20,9:1},{0:18,3:31,4:17},{1:19,2:31,5:32,6:33},{2:17,7:16},
            {3:32,8:36},{1:20,3:33,8:34,9:21},{4:16,8:30,11:15},{5:36,6:34,7:30,11:29,12:24,17:26,18:25,20:27},{1:1,6:21,10:2,12:35,13:22},
            {9:2,13:3},{7:15,8:29,14:14,20:28},{8:24,9:35,13:4,19:5,21:23},{9:22,10:3,12:4},{11:14,15:13},
            {14:13,16:12},{15:12,17:11},{8:26,16:11,18:10},{8:25,17:10,21:9},{12:5,23:6},
            {8:27,11:28},{12:23,18:9,22:8},{21:8,23:7},{19:6,22:7}]
CAPACITY = [110,140,110,110,110,140,110,110,110,140,
            140,140,110,110,140,140,140,110,110,140,
            140,140,110,140,140,140,140,110,110,140,
            140,110,140,110,140,140,140]

class TestEnv(object):
    def __init__(self, num_v, num_e, topo, max_hop, ttl,sparsity=0.3, dm_seq_len=1000):
        self.num_v = num_v
        self.num_e = num_e
        self.topo = topo
        self.sparsity = sparsity
        self.dm_seq_len = dm_seq_len
        self.max_hop = max_hop 
        self.gamma = 2
        self.ttl = ttl
        self.generate_fixed_dm()

    def fetch_dm(self, no):
        assert (no>=0 and no<self.dm_seq_len), "Testing terminated already."
        return self.fixed_dm[:,:,no]
    
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

    def learningtoroute(self, split_weight, dm, capa_mat):
        utilization = np.zeros(self.num_e)
        capacity_vec = capacity_mat2vec(capa_mat,self.num_v,self.num_e,self.topo)
        split_mat = np.zeros((self.num_v,self.num_v,self.num_v))
        for dst in range(self.num_v):
            for src in range(self.num_v):
                if src==dst:
                    split_mat[dst,src,dst] = 1
                else:    
                    total_weight = 0
                    for neighbor, link in self.topo[src].items():
                        split_portion = np.exp(-self.gamma*split_weight[dst,link])
                        total_weight += split_portion
                        split_mat[dst,src,neighbor] = split_portion
                    split_mat[dst,src,:] = split_mat[dst,src,:]/total_weight
        total_arrived = 0
        for dst in range(self.num_v):
            split_mat_dst = split_mat[dst,:,:]
            for src in range(self.num_v):
                demand_sd = dm[src,dst]
                if demand_sd != 0:
                    t = 0
                    util_flow = np.zeros(self.num_e)
                    F_prev = np.zeros(self.num_v)
                    f_prev = np.zeros(self.num_v)
                    F_prev[src] = 1.0 
                    f_prev[src] = 1.0
                    while t < self.ttl:
                        F_t, util_plus = spread_traffic_step(self.num_v,self.num_e,self.topo,src,dst,split_mat_dst,F_prev)
                        f_t = spread_traffic_flowt0(src,dst,split_mat_dst,f_prev)
                        util_flow += util_plus
                        F_prev = F_t
                        f_prev = f_t
                        t += 1
                    arrived = f_t[dst]
                    total_arrived += arrived
                    utilization += util_flow/(self.ttl+1)
        mlu = np.max(utilization)
        util_cap_dif = np.clip(utilization-capacity_vec,0,mlu)
        return mlu, np.max(util_cap_dif), total_arrived

    def baseline1(self, dm, capa_mat):
        utilization = np.zeros(self.num_e)
        capacity_vec = capacity_mat2vec(capa_mat,self.num_v,self.num_e,self.topo)
        shortest_path = self.shortest_path_cal()
        for src in range(self.num_v):
            for dst in range(self.num_v):
                demand_flow = dm[src,dst]
                if demand_flow != 0:
                    path = shortest_path[src][dst]
                    n_prev = src
                    for hop in range(len(path)):
                        n_next = path[hop]
                        link = self.topo[n_prev][n_next]
                        utilization[link] += demand_flow
                        n_prev = n_next
        util_cap_dif = np.clip(utilization-capacity_vec,0,1e20)
        return np.max(utilization), np.max(util_cap_dif)

    def baseline2(self, dm, capa_mat):
        utilization = np.zeros(self.num_e)
        capacity_vec = capacity_mat2vec(capa_mat,self.num_v,self.num_e,self.topo)
        shortest_path = self.shortest_path_cal(capa_mat=capa_mat)
        for src in range(self.num_v):
            for dst in range(self.num_v):
                demand_flow = dm[src,dst]
                if demand_flow != 0:
                    path = shortest_path[src][dst]
                    n_prev = src
                    #print('path',path)
                    for hop in range(len(path)):
                        n_next = path[hop]
                        link = self.topo[n_prev][n_next]
                        utilization[link] += demand_flow
                        n_prev = n_next
        util_cap_dif = np.clip(utilization-capacity_vec,0,1e20)
        return np.max(utilization), np.max(util_cap_dif) 

    def shortest_path_cal(self,capa_mat=np.array([])):
        
        num_v = self.num_v
        topo = self.topo
        distance_graph = np.ones((num_v,num_v))*float('inf')
        if capa_mat.shape[0] == 0:
            for i in range(num_v):
                distance_graph[i,i] = 0
                for neighbor, _ in topo[i].items():
                    distance_graph[i,neighbor] = 1
        else:
            for i in range(num_v):
                distance_graph[i,i] = 0
                for neighbor, _ in topo[i].items():
                    if capa_mat[i,neighbor] > 1.0:
                        distance_graph[i,neighbor] = 1/capa_mat[i,neighbor]
        
        path = dict()
        for src in range(num_v):
            graph = distance_graph
            visited = [src]
            path[src] = {src:[]}
            vertices = [i for i in range(num_v)]
            vertices.remove(src)
            pre = next = src

            while vertices:
                distance = float('inf')
                for v in visited:
                    for d in vertices:
                        new_dist = graph[src, v] + graph[v, d]
                        if new_dist <= distance and d in self.topo[v]:
                            distance = new_dist
                            next = d
                            pre = v
                            graph[src, d] = new_dist
                path[src][next] = [i for i in path[src][pre]]
                path[src][next].append(next)

                visited.append(next)
                vertices.remove(next)
        return path
    
    def ALTRoutingEva(self,action,dm,capacity):
        
        capacity_vec = capacity_mat2vec(capacity,self.num_v,self.num_e,self.topo)
        self.altitude = naive_altitude(self.num_v,self.topo)
        modf = np.zeros((self.num_v,self.num_v))
        for i in range(self.num_v-1):
            modf[i+1,:(i+1)] = action[i+1,:(i+1)]
            modf[i,(i+1):] = action[i,i:(self.num_v-1)]
        modf_norm = np.zeros_like(modf)
        modf_norm = (self.num_v/2-1)*modf*modf*modf
        #corr_range_mat = self.corr_range * np.ones_like(modf_norm)
        #upper = np.clip(modf_norm-corr_range_mat,0,self.num_v)
        #lower = np.clip(-corr_range_mat-modf_norm,0,self.num_v)
        #exceeded_up = upper>lower
        #exceeded_lo = upper<=lower
        #exceeded = exceeded_up * upper + exceeded_lo * lower
        #exceeded_punish = np.mean(exceeded)
        #modf_norm = np.clip(modf_norm,-self.corr_range,self.corr_range)
        self.altitude = self.altitude + modf_norm
        split_weight = action[:,(self.num_v-1):]
        demand = dm
        self.traffic = {}
        total_arrived = 0
        self.exceed_max_search = False
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
                    util_split, arrived = self.spread_traffic(src,dst,split_weight[dst])
                utilization += demand[src,dst]*util_split
                total_arrived += arrived
                if self.exceed_max_search:
                    print("[flow ({0},{1})] arrived:{2}".format(src,dst,arrived))
        mlu = np.max(utilization)
        util_cap_dif = np.clip(utilization-capacity_vec,0,mlu)
        return mlu, total_arrived, np.max(util_cap_dif)

    def spread_traffic(self, src, dst, link_weight):

        if src == dst:
            self.traffic[(src,dst)]=(np.zeros(self.num_e),1.0)
            return np.zeros(self.num_e), 1.0
        elif self.exceed_max_search:
            return np.zeros(self.num_e), 0.0
        else:
            self.hop_count += 1
            total_weight = 0.0
            util = np.zeros(self.num_e)
            arrived_flow = 0.0
            for neighbor, link in self.topo[src].items():
                if(self.hop_count>=self.max_hop):
                    self.exceed_max_search = True
                if self.altitude[dst,neighbor] <= self.altitude[dst,src]:
                    continue
                split_portion = np.exp(-self.gamma*link_weight[link])
                total_weight += split_portion
                if (neighbor,dst) in self.traffic:
                    util_plus, arrived_plus = self.traffic[(neighbor,dst)]
                else:
                    util_plus, arrived_plus = self.spread_traffic(neighbor, dst, link_weight)
                util[link] += split_portion
                util += split_portion * util_plus
                arrived_flow += split_portion * arrived_plus
            if total_weight == 0:
                self.traffic[(src,dst)]=(np.zeros(self.num_e),0.0)
                return np.zeros(self.num_e), 0.0
            util = util/total_weight
            arrived_flow = arrived_flow/total_weight
            if (src,dst) not in self.traffic:
                self.traffic[(src,dst)]=(util,arrived_flow)

            return util, arrived_flow

    def naive(self,dm):

        self.altitude = naive_altitude(self.num_v,self.topo)
        split_weight = np.ones((self.num_v,self.num_e))
        demand = dm
        self.traffic = {}
        total_arrived = 0
        self.exceed_max_search = False
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
                    util_split, arrived = self.spread_traffic(src,dst,split_weight[dst])
                utilization += demand[src,dst]*util_split
                total_arrived += arrived
        mlu = np.max(utilization)
        return mlu, total_arrived

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

def spread_traffic_step(num_v,num_e,topo,src,dst,split_mat,F_t_minus_1):
    util_plus = np.zeros(num_e)
    for node in range(num_v):
        for neighbor,link in topo[node].items():
            util_plus[link] += F_t_minus_1[node]*split_mat[node,neighbor]
    F_t = np.matmul(split_mat.T,F_t_minus_1)
    F_t[src] += 1
    return F_t, util_plus

def spread_traffic_flowt0(src,dst,split_mat,f_t_minus_1):
    f_t = np.matmul(split_mat.T,f_t_minus_1)
    return f_t

def state_trans4lr(prev_obs,dm):
    # get next state
    state = np.roll(prev_obs, -1, axis=-1)
    state[:,:,-1] = dm
    return state

def main():
    test = TestEnv(NUM_VERTICE,NUM_EDGE,TOPOLOGY,MAX_HOP,64)
    model = PPO2.load(MODEL_NAME)
    net_env = env.ALTEnv(NUM_VERTICE, NUM_EDGE, TOPOLOGY, CAPACITY,
                        k=S_LEN, ep_length=EP_LENGTH, max_hop=MAX_HOP)

    model_base = PPO2.load(MODEL_NAME_BASE)

    ## evaluate the agent
    #mean_reward, n_steps = evaluate_policy(model, net_env, n_eval_episodes=10)
    #print("mean reward",mean_reward)
    #print("n_steps", n_steps)

    obs = net_env.reset()
    dms = []
    mlu_dijkstra1 = []
    mlu_dijkstra2 = []
    mlu_l = []
    mlu_alt = []
    arrived_l = []
    arrived_alt = []
    util_exd_dij1 = []
    util_exd_dij2 = []
    util_exd_alt = []
    util_exd_l = []
    state_lr = np.zeros((NUM_VERTICE,NUM_VERTICE,10))
    for _ in range(10000):
        action, _states = model.predict(obs)
        action_mat = action.reshape((NUM_VERTICE,NUM_VERTICE-1+NUM_EDGE))
        dm = obs[:,:,0]
        capa_mat = obs[:,:,1]
        state_lr = state_trans4lr(state_lr,dm)
        action_lr = model_base.predict(state_lr)
        action_lr_mat = action_lr[0].reshape((NUM_VERTICE,NUM_EDGE))
        b1_mlu, b1_utilexd = test.baseline1(dm,capa_mat)
        b2_mlu, b2_utilexd = test.baseline2(dm,capa_mat)
        l_mlu, l_utilexd, l_arrived = test.learningtoroute(action_lr_mat,dm,capa_mat)
        a_mlu, a_arr, a_utilexd = test.ALTRoutingEva(action_mat,dm,capa_mat)

        dm_sum = np.sum(dm)
        mlu_dijkstra1.append(b1_mlu)
        mlu_dijkstra2.append(b2_mlu)
        mlu_l.append(l_mlu)
        mlu_alt.append(a_mlu)
        util_exd_dij1.append(b1_utilexd)
        util_exd_dij2.append(b2_utilexd)
        util_exd_l.append(l_utilexd)
        util_exd_alt.append(a_utilexd)
        arrived_l.append(l_arrived)
        arrived_alt.append(a_arr)
        dms.append(dm_sum)
        obs, _, _, _ = net_env.step(action)
    
    print("MEAN: mlu_d1 {0} mlu_d2 {1} mlu_lr {2} mlu_alt {3} utilexd: d1 {4} d2 {5} lr {6} alt {7} arrived: lr {8} alt {9}\n".format(
        np.mean(mlu_dijkstra1),np.mean(mlu_dijkstra2),np.mean(mlu_l),np.mean(mlu_alt),
        np.mean(util_exd_dij1),np.mean(util_exd_dij2),np.mean(util_exd_l),np.mean(util_exd_alt),
        np.mean(arrived_l),np.mean(arrived_alt)
    ))   
    print("STD: mlu_d1 {0} mlu_d2 {1} mlu_lr {2} mlu_alt {3} utilexd: d1 {4} d2 {5} lr {6} alt {7} arrived: lr {8} alt {9}\n".format(
        np.std(mlu_dijkstra1),np.std(mlu_dijkstra2),np.std(mlu_l),np.std(mlu_alt),
        np.std(util_exd_dij1),np.std(util_exd_dij2),np.std(util_exd_l),np.std(util_exd_alt),
        np.std(arrived_l),np.std(arrived_alt)
    ))   
    print("DM mean {0} std {1}".format(np.mean(dms),np.std(dms)))
    """
    print("MEAN: mlu_d1 {0} mlu_d2 {1} mlu_alt {2} utilexd: d1 {3} d2 {4} alt {5} arrived: alt {6}\n".format(
        np.mean(mlu_dijkstra1),np.mean(mlu_dijkstra2),np.mean(mlu_alt),
        np.mean(util_exd_dij1),np.mean(util_exd_dij2),np.mean(util_exd_alt),
        np.mean(arrived_alt)
    ))
    print("STD: mlu_d1 {0} mlu_d2 {1} mlu_alt {2} utilexd: d1 {3} d2 {4} alt {5} arrived: alt {6}\n".format(
        np.std(mlu_dijkstra1),np.std(mlu_dijkstra2),np.std(mlu_alt),
        np.std(util_exd_dij1),np.std(util_exd_dij2),np.std(util_exd_alt),
        np.std(arrived_alt)
    ))
    print("DM mean {0} std {1}".format(np.mean(dms),np.std(dms)))
    """

if __name__ == "__main__":
    main()