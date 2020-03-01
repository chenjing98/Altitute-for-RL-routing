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
    def __init__(self,sess,ob_space,ac_space,n_env,n_steps,n_batch,input_dim, hid_dims, out_dim, reuse=False,**kwargs):
        super(GnnPolicy, self).__init__(sess,ob_space,ac_space,n_env,n_steps,n_batch,reuse=reuse,scale=True)

        with tf.variable_scope("GnnModel", reuse=reuse):
            self.act_fn = tf.nn.relu

            self.gnn_weights, self.gnn_bias = self.gnn_para_init()

            extracted_features = gnn_forward(self.processed_obs, **kwargs)
            extracted_features = tf.layers.flatten(extracted_features)


    def gnn_para_init(self, input_dim, hid_dims, out_dim, dropout, scope):
    
        weights = []
        bias = []

        curr_in_dim = E.shape[-1]

        # hidden layers
        for hid_dim in hid_dims:
          weights.append(glorot([curr_in_dim, hid_dim], scope=scope))
          bias.append(zeros([hid_dim], scope=scope))
          curr_in_dim = hid_dim

        return weights, bias

    def gnn_forward(self, input):

        E = input # edge features matrix [N x N x 2]
        num_n = E.shape[0]
        P = E.shape[-1]

        X = tf.ones([num_n, P], tf.float32) # node feature matrix [N x 2]

        y = X 
        for l in range(len(self.gnn_weights)):
          y = tf.matmul(y, self.gnn_weights[l])
          y += self.gnn_bias[l]
          y = tf.matmul(tf.transpose(E,[2,0,1]), y)
          y = tf.sigmoid(y)

        return y


def glorot(shape, dtype=tf.float32, scope='default'):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    with tf.variable_scope(scope):
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        init = tf.random_uniform(
            shape, minval=-init_range, maxval=init_range, dtype=dtype)
        return tf.Variable(init)


def zeros(shape, dtype=tf.float32, scope='default'):
    with tf.variable_scope(scope):
        init = tf.zeros(shape, dtype=dtype)
        return tf.Variable(init)