import argparse

def save_args(args, fn):
    with open(fn, 'w') as fw:
        for items in vars(args):
            fw.write('%s %s\n' % (items, vars(args)[items]))


def load_args():
    parser = argparse.ArgumentParser(description="Config to train the GestSync model")

    # Training params
    parser.add_argument('--num_frames', default=25, type=int, required=False, help="Number of frames in a temporal window")
    parser.add_argument('--total_frames', default=175, type=int, required=False, help="Total number of frames to get both positive and negative samples")
    parser.add_argument('--similarity_type', default="mean", required=False, help="Spatial similarity to be used [mean or max] for keypoint-vector representation model")
        
    # Checkpoint params
    parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='checkpoints', required=False, type=str)
    parser.add_argument('--checkpoint_path', help='Resume the model from this checkpoint', default=None, type=str)

    # Frame params
    parser.add_argument('--height', type=int, default=270)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--fps', type=int, default=25)
    
    # Audio params
    parser.add_argument('--sample_rate', type=int, default=16000)
    
    # General
    parser.add_argument('--learning_rate', default=1e-4, required=False, help="Learning rate for the optimizer")
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data workers')
    parser.add_argument('--ckpt_interval', type=int, default=1000, help='Steps to save the trained model')
    parser.add_argument('--val_interval', type=int, default=3000, help='Steps to validate the model')
    parser.add_argument('--num_epochs', type=int, default=100, required=False, help='Number of epochs to train the model')    
    parser.add_argument('--n_gpu', type=int, default=1, required=False, help='Number of GPUs for non-distributed training')

    # Data-path params
    parser.add_argument('--data_path_videos', type=str, default="preprocess/preprocessed_videos")
    parser.add_argument('--data_path_kps', type=str, default="preprocess/keypoints")    

    # Distributed training
    parser.add_argument('--use_dist_training', default=False, type=str, help="Use distributed training with multiple GPUs for faster training")
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank of node')
    parser.add_argument('--sync_bn', default=True)

    args = parser.parse_args()

    return args
