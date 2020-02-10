import os
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env

import Env_base as env
from gnnpolicy import MlpPolicy_3l

S_LEN = 10
EP_LENGTH = 50
TTL = 64
"""
# NSFNet
NUM_VERTICE = 14
NUM_EDGE = 21
GRAPH = [{1:1,2:2,3:3},{0:1,2:4,7:7},{0:2,1:4,5:5},{0:3,4:6,8:9},{3:6,5:8,6:0},{2:5,4:8,12:18,13:19},\
         {4:0,7:10},{1:7,6:10,10:11},{3:9,9:13,11:14},{8:13,10:20,12:16},{7:11,9:20,11:12,13:17},\
         {8:14,10:12,12:15},{5:18,9:16,11:15},{5:19,10:17}] # NSFNet topology
CAPACITY = [110,110,110,110,110,110,140,110,140,110,110,140,110,110,110,110,110,110,110,110,110] # kbps
"""
# GEANT2 topology
NUM_VERTICE = 24
NUM_EDGE = 37
GRAPH = [{1:0,2:18},{0:0,3:19,6:20,9:1},{0:18,3:31,4:17},{1:19,2:31,5:32,6:33},{2:17,7:16},
            {3:32,8:36},{1:20,3:33,8:34,9:21},{4:16,8:30,11:15},{5:36,6:34,7:30,11:29,12:24,17:26,18:25,20:27},{1:1,6:21,10:2,12:35,13:22},
            {9:2,13:3},{7:15,8:29,14:14,20:28},{8:24,9:35,13:4,19:5,21:23},{9:22,10:3,12:4},{11:14,15:13},
            {14:13,16:12},{15:12,17:11},{8:26,16:11,18:10},{8:25,17:10,21:9},{12:5,23:6},
            {8:27,11:28},{12:23,18:9,22:8},{21:8,23:7},{19:6,22:7}]
CAPACITY = [110,140,110,110,110,140,110,110,110,140,
            140,140,110,110,140,140,140,110,110,140,
            140,140,110,140,140,140,140,110,110,140,
            140,110,140,110,140,140,140]
#NUM_VERTICE = 6
#NUM_EDGE = 9
#GRAPH = [{1:1,5:2},{0:1,5:3,3:5,2:4},{1:4,3:6},{2:6,1:5,5:7,4:8},{3:8,5:0},{0:2,1:3,3:7,4:0}]
MODEL_NAME = "ppo2_base_geant"
MODEL_NAME_NEW = "ppo2_base_geant2"
TENSORBOARD_LOG_DIR = "./tensorlog_base"

def vec_fn():
    net_env = env.basicEnv(NUM_VERTICE, NUM_EDGE, GRAPH, CAPACITY,
                           k=S_LEN, ep_length=EP_LENGTH, ttl=TTL)
    return net_env

def main():

    net_env = env.basicEnv(NUM_VERTICE, NUM_EDGE, GRAPH, CAPACITY,
                           k=S_LEN, ep_length=EP_LENGTH, ttl=TTL)

    vec_netenv = DummyVecEnv([vec_fn])
    # check environment
    #check_env(net_env)
    if not os.path.exists(TENSORBOARD_LOG_DIR):
        os.makedirs(TENSORBOARD_LOG_DIR)

    #model = PPO2(MlpPolicy, net_env,
    #             verbose=1,
    #             tensorboard_log=TENSORBOARD_LOG_DIR)

    #model = PPO2(MlpPolicy_3l, net_env,
    #             verbose=1,
    #             tensorboard_log=TENSORBOARD_LOG_DIR)
    
    model = PPO2.load(MODEL_NAME,env=vec_netenv)
    print("Model built.")
    model.learn(total_timesteps=100000)
    print("Training terminated.")

    # save model
    model.save(MODEL_NAME_NEW)

    obs = net_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, _, _ = net_env.step(action)
        print("[Epoch {0}] [reward {1}]".format(i,rewards))

if __name__ == "__main__":
    main()