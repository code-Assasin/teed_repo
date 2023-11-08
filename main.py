import argparse
import wandb
import sys

sys.path.append("../")
import data
from teed_trainer import TEEDTrainer
import configs

# argparse to get params
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--noise_std", type=float, default=0.1)
parser.add_argument("--load_checkpoint", type=bool, default=False)
# parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--dataset", type=str, default="cifar10")

# convert to params
params = parser.parse_args()
params = vars(params)

config_wandb = dict(defense=params)
run_name = 'trial'
# save_name = save_name_generator(param)
# param["save_name"] = save_name
params["checkpoint_path"] = configs.directory_names["save_dir"]

logger = wandb.init(
    entity=configs.wandb_config["entity"],
    project=configs.wandb_config["project"],
    reinit=configs.wandb_config["reinit"],
    name=run_name,
    config=config_wandb,
)
params["logger"] = logger

# get dataset
data_loading = data.DataLoading(params=params)
trainset, trainloader, testset, testloader = data_loading.get_data()

params["trainloader"] = trainloader
params["testloader"] = testloader

# Train model
trainer = TEEDTrainer(params=params, device="cuda")
trainer.solve()


