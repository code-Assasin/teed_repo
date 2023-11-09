CUDA_VISIBLE_DEVICES=3 python main.py --batch_size 64 --run_name scheduler_cosine_0.01 --lr 0.01 --scheduler cosine
CUDA_VISIBLE_DEVICES=3 python main.py --batch_size 64 --run_name scheduler_cosine_0.001 --lr 0.001 --scheduler cosine

CUDA_VISIBLE_DEVICES=3 python main.py --batch_size 64 --run_name scheduler_constant_0.1 --lr 0.1 --scheduler constant
CUDA_VISIBLE_DEVICES=3 python main.py --batch_size 64 --run_name scheduler_constant_0.01 --lr 0.01 --scheduler constant
