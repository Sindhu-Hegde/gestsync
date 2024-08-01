import os, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import subprocess
# import audio.hparams as hp 
# import audio.audio_utils as audio
import librosa
import cv2
import numpy as np

#=========  DDP utils from https://github.com/pytorch/vision/blob/4cbe71401fc6e330e4c4fb40d47e814911e63399/references/video_classification/utils.py 

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
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


def save_on_master(state, save_path):
    if is_main_process():
        # print('Saving checkpoint: {}'.format(save_path))
        torch.save(state, save_path)


def init_distributed_mode(args):

    args.distributed = False

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # -- job started with torch.distributed.launch

        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.global_rank = int(os.environ['RANK'])
        args.n_gpu_per_node = torch.cuda.device_count()

        args.job_id = args.global_rank 
        args.device = args.local_rank 

        args.n_nodes = args.world_size // args.n_gpu_per_node
        args.node_id = args.global_rank // args.n_gpu_per_node

        args.distributed = True


    elif 'SLURM_PROCID' in os.environ:
        # -- SLURM JOB
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        print (args.gpu)

        # # job ID
        args.job_id = os.environ['SLURM_JOB_ID']

        args.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        args.node_id = int(os.environ['SLURM_NODEID'])

        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.global_rank = int(os.environ['SLURM_PROCID'])

        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.n_gpu_per_node = args.world_size // args.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames',
            os.environ['SLURM_JOB_NODELIST']])
        args.master_addr = hostnames.split()[0].decode('utf-8')
        args.master_port = 19500
        assert 10001 <= args.master_port <= 20000 or args.world_size == 1
        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = str(args.master_addr)
        os.environ['MASTER_PORT'] = str(args.master_port)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.global_rank)


        if args.world_size >= 1:
            args.distributed = True
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(args.gpu)

    if args.distributed:
        args.dist_backend = 'nccl'
        print('| distributed init (rank {}, world_size {}): {}'.format(
            args.rank, args.world_size, args.dist_url), flush=True)
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                            world_size=args.world_size, rank=args.rank)
        setup_for_distributed(args.rank == 0)

#================================================================================================


