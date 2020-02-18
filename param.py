import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--topology", type=str, default="nsf", help="network topology, options: nsf, geant2,mini")
parser.add_argument("--n_layer", type=int, default=3, help="number of layers of the policy network")

parser.add_argument("--model_exists", type=bool, default=False, help="whether to train with an existing model")
parser.add_argument("--ex_model_name", type=str, default=None, help="the existed model name")
parser.add_argument("--model_name", type=str, default="base", help="the model name for saving parameters")
parser.add_argument("--tb_log_dir", type=str, default="./tensorlog", help="where to store tensorboard logs")

parser.add_argument("--train_timesteps", type=int, default=100000, help="the total timesteps of training")

args = parser.parse_args()