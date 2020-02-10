import gym
import tensorflow as tf

from stable_baselines.common.policies import FeedForwardPolicy

class MlpPolicy_3l(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MlpPolicy_3l, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[64,64,64],
                                                          vf=[64,64,64])],
                                           feature_extraction="mlp")


class GnnPolicy(FeedForwardPolicy):
    def __init__(self,sess,ob_space,ac_space,n_env,n_steps,n_batch,reuse=False,**kwargs):
        super(GnnPolicy, self).__init__(sess,ob_space,ac_space,n_env,n_steps,n_batch,reuse=reuse,scale=True)

        with tf.variable_scope("GnnModel", reuse=reuse):
            activ = tf.nn.relu

            extracted_features = gnn_net(self.processed_obs, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)


def gnn_net(graph,capacity,failure_prob):
    
    return 1
