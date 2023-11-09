CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 64 --run_name scheduler_one_cycle_0.01 --scheduler onecycle --lr 0.01
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 64 --run_name scheduler_one_cycle_0.001 --scheduler onecycle --lr 0.001
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size 64 --run_name scheduler_one_cycle_0.1 --scheduler onecycle --lr 0.1
