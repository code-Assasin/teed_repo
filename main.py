import argparse
import wandb
import sys

sys.path.append("../")
import data
from trainers.teed_trainer import TEEDTrainer
from trainers.pix2pix_trainer import Pix2PixTrainer
from utils.misc import *
import configs

# argparse to get params
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--noise_std", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--scheduler", choices=["cosine", "multistep", "onecycle", "constant"], default="multistep")
parser.add_argument("--edge_model", choices=["teed", "pix2pix"], default="teed")
# convert to params
params = parser.parse_args()
params = vars(params)

config_wandb = dict(defense=params)

if params["run_name"] is None:
    print("run_name is None")
    run_name = run_name_generator(params)
    params["run_name"] = run_name

params["checkpoint_path"] = save_path_generator(
    params, save_dir=configs.directory_names["save_dir"]
)

logger = wandb.init(
    entity=configs.wandb_config["entity"],
    project=configs.wandb_config["project"],
    reinit=configs.wandb_config["reinit"],
    name=params["run_name"],
    config=config_wandb,
) # settings is to prevent wandb from streaming file chunk errors
params["logger"] = logger

# get dataset
data_loading = data.DataLoading(params=params)
trainset, trainloader, testset, testloader = data_loading.get_data()

params["trainloader"] = trainloader
params["testloader"] = testloader

# Train model
print("Training model")
trainer = Pix2PixTrainer(params=params, device="cuda")
trainer.train_one_epoch(0)  # dummy call to initialize the model
exit()
trainer.solve()
