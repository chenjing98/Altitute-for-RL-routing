import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--topology", type=str, default="nsf", help="network topology, options: nsf, geant2,mini")
parser.add_argument("--n_layer", type=int, default=3, help="number of layers of the policy network")

parser.add_argument("--model_exists", type=bool, default=False, help="whether to train with an existing model")
parser.add_argument("--ex_model_name", type=str, default=None, help="the existed model name")
parser.add_argument("--model_name", type=str, default="ppo2", help="the model name for saving parameters (default: ppo2)")
parser.add_argument("--tb_log_dir", type=str, default="./tensorlog", help="where to store tensorboard logs (default:./tensorlog)")

parser.add_argument("--train_timesteps", type=int, default=1000000, help="the total timesteps of training (default: 1000000)")

parser.add_argument("--ttl", type=int, default=64, help="TTL for Learning To Route simulation (default: 64)")

args = parser.parse_args()