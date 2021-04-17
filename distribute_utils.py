# -*- coding:utf-8 -*-
import os

import torch
import torch.distributed as dist


# python -m torch.distributed.launch --nproc_per_node=4 --use_env train_multi_gpu_using_launch.py
# --use_env 传入时，有以下参数
def init_distributed_mode(args):
    # 多机多卡时，world_size表示使用几台机器，rank是第几台机器 local_rank表示第几个gpu
    # 单机多卡时，world_size表示有几个gpu，rank表示第几个gpu local_rank = rank
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print("-----------------------------------------")
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)  # 对当前线程指定gpu 多卡时，是对每个gpu起一个进程
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    # 创建进程组
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                            rank=args.rank)
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)  # 对不同设备下value进行求和操作
        if average:
            value /= world_size

        return value


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
