CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --run_name scheduler_multistep_0.01 --lr 0.01 --scheduler multistep
CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --run_name scheduler_multistep_0.001 --lr 0.001 --scheduler multistep

CUDA_VISIBLE_DEVICES=2 python main.py --batch_size 64 --run_name scheduler_constant_0.001 --lr 0.001 --scheduler constant