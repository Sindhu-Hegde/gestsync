import os, subprocess, argparse
import numpy as np

from tqdm import tqdm
from glob import glob

from dataset import *
from sync_models.gestsync_models import *

from config import *
from utils.audio_utils import *
from utils.dist_utils import *

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.distributed import DistributedSampler


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# Initialize global variables
global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()

# Training with FP16
scaler = torch.cuda.amp.GradScaler()


def accuracy(output, target, topk=(1, 2, 3)):
	
	'''
	This function computes the accuracy@k for the specified values of k
	Args:
		- output: the output of the model
		- target: the target of the model
		- topk: the values of k to compute the accuracy@k
	Returns:
		- res: the accuracy@k for the specified values of k
	'''

	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))

	return res


def get_sync_loss(audio_emb, video_emb):

	'''
	This function computes the sync-loss using cosine similarity
	Args:
		- audio_emb: the audio embedding
		- video_emb: the video embedding
	Returns:
		- loss_row: the loss for the rows
		- loss_col: the loss for the columns
		- a1_row: the accuracy@1 for the rows
		- a2_row: the accuracy@2 for the rows
		- a3_row: the accuracy@3 for the rows
		- a1_col: the accuracy@1 for the columns
		- a2_col: the accuracy@2 for the columns
		- a3_col: the accuracy@3 for the columns
	'''


	time_size = audio_emb.size(2)
	losses_row = []
	losses_col = []
	a1s_row, a2s_row, a3s_row = [], [], []
	a1s_col, a2s_col, a3s_col = [], [], []
	label = torch.arange(time_size).cuda()
	for i in range(audio_emb.size(0)):
		ft_v = video_emb[[i],:,:].transpose(2,0)
		ft_a = audio_emb[[i],:,:].transpose(2,0)
		
		sim = cos(ft_v.expand(-1, -1, time_size), ft_a.expand(-1, -1, time_size).transpose(0, 2)) 
		
		if args.use_dist_training:
			output = model.module.logits_scale(sim.unsqueeze(-1)).squeeze(-1)
		else:
			output = model.logits_scale(sim.unsqueeze(-1)).squeeze(-1)
		
		losses_row.append(logloss(output, label))
		a1, a2, a3 = accuracy(output, label)
		a1s_row.append(a1.item())
		a2s_row.append(a2.item())
		a3s_row.append(a3.item())

		losses_col.append(logloss(output.T, label))
		a1, a2, a3 = accuracy(output.T, label)
		a1s_col.append(a1.item())
		a2s_col.append(a2.item())
		a3s_col.append(a3.item())

	loss_row = sum(losses_row) / len(losses_row)
	a1_row = sum(a1s_row) / len(a1s_row)
	a2_row = sum(a2s_row) / len(a2s_row)
	a3_row = sum(a3s_row) / len(a3s_row)

	loss_col = sum(losses_col) / len(losses_col)
	a1_col = sum(a1s_col) / len(a1s_col)
	a2_col = sum(a2s_col) / len(a2s_col)
	a3_col = sum(a3s_col) / len(a3s_col)

	return loss_row, a1_row, a2_row, a3_row, loss_col, a1_col, a2_col, a3_col



def train(model, train_data_loader, val_data_loader, optimizer, checkpoint_dir, checkpoint_interval, validation_interval, nepochs):

	'''
	This function trains the model
	Args:
		- model: the model to train
		- train_data_loader: the training data loader
		- val_data_loader: the validation data loader
		- optimizer: the optimizer
		- checkpoint_dir: the directory to save the checkpoints
		- checkpoint_interval: the interval to save the checkpoints
		- validation_interval: the interval to validate the model
		- nepochs: the number of epochs to train the model
	'''

	global global_step, global_epoch
	resumed_step = global_step
	
	while global_epoch < nepochs:

		print("Epoch: ", global_epoch)
		running_loss = 0.
		running_loss_row = 0.
		running_loss_col = 0.
		running_a1_row = 0.
		running_a2_row = 0.
		running_a3_row = 0.
		running_a1_col = 0.
		running_a2_col = 0.
		running_a3_col = 0.
		
		prog_bar = tqdm(enumerate(train_data_loader))
		
		for step, batch_sample in prog_bar:

			if batch_sample == 0:
				continue
			
			model.train()
			optimizer.zero_grad()

			video = batch_sample['video']									# Bx3xT_vid_totalx270x480
			audio = batch_sample['audio']  									# BxT_aud_total

			# Extract log-mel filterbanks  
			mel, _, _, _ = wav2filterbanks(audio.cuda())					# BxT_spec_totalx80

			# Get the temporal window of 1-second
			mel_chunk = torch.split(mel, split_size_or_sections=num_frames*4, dim=1)
			mel_chunk = torch.stack(mel_chunk, dim=1)  
			mel_chunk = mel_chunk.permute(0,1,3,2)[:, None]					# Bx1xTx80x100

			vid_chunk = torch.split(video, split_size_or_sections=num_frames, dim=2)
			vid_chunk = torch.stack(vid_chunk, dim=2)  						# Bx3xTx25x270x480	

			# To get the embeddings, concat all the temporal windows to the batch dimension
			B=vid_chunk.size(0)
			face_sequences = torch.cat([vid_chunk[:, :, i] for i in range(vid_chunk.size(2))], dim=0)
			audio_sequences = torch.cat([mel_chunk[:, :, i] for i in range(mel_chunk.size(2))], dim=0)
			# print("Model input - video: ", face_sequences.shape)			# (B*T)x3x25x270x80
			# print("Model input - audio: ", audio_sequences.shape)			# (B*T)x1x80x100


			with torch.cuda.amp.autocast():

				# Extract the embeddings
				if args.use_dist_training:
					video_embedding, video_feat = model.module.forward_vid(face_sequences.cuda(), return_feats=True)
					video_embedding = torch.mean(video_embedding, axis=-1).unsqueeze(-1)
					audio_embedding = model.module.forward_aud(audio_sequences.cuda())
				else:
					video_embedding, video_feat = model.forward_vid(face_sequences.cuda(), return_feats=True)
					video_embedding = torch.mean(video_embedding, axis=-1).unsqueeze(-1)
					audio_embedding = model.forward_aud(audio_sequences.cuda())


				# Get the temporal dimension back from batch dimension
				audio_embedding = torch.split(audio_embedding, B, dim=0) 						
				audio_embedding = torch.stack(audio_embedding, dim=2) 							
				audio_embedding = audio_embedding.squeeze(3) 						

				video_embedding = torch.split(video_embedding, B, dim=0) 					
				video_embedding = torch.stack(video_embedding, dim=2) 					
				video_embedding = video_embedding.squeeze(3)								

				# print("Video embedding: ", video_embedding.shape)						# Bx1024xT
				# print("Audio embedding: ", audio_embedding.shape)						# Bx1024xT

				# Compute the loss and accuracies@k
				loss_row, a1_row, a2_row, a3_row, loss_col, a1_col, a2_col, a3_col = get_sync_loss(audio_embedding, video_embedding)

			# Backpropagate the loss
			loss = loss_row + loss_col
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()

			# Save the model at desired intervals
			if global_step == 0 or global_step % checkpoint_interval == 0:
				save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)

			# Validate the model at desired intervals
			if global_step % validation_interval == 0:
				with torch.no_grad():
					validate_model(val_data_loader, model)


			# Update training logs
			global_step += 1
			running_loss += loss.item()
			running_loss_row += loss_row.item()
			running_loss_col += loss_col.item()
			running_a1_row += a1_row
			running_a2_row += a2_row
			running_a3_row += a3_row
			running_a1_col += a1_col
			running_a2_col += a2_col
			running_a3_col += a3_col

			prog_bar.set_description('Total loss: {:.3f} | Loss-r: {:.3f}, Loss-c: {:.3f} | A1r: {:.2f}, A2r: {:.2f}, A3r: {:.2f} | A1c: {:.2f}, A2c: {:.2f}, A3c: {:.2f}'
																.format(running_loss / (step + 1),
																		running_loss_row / (step + 1),
																		running_loss_col / (step + 1),
																		running_a1_row / (step + 1),
																		running_a2_row / (step + 1),
																		running_a3_row / (step + 1),
																		running_a1_col / (step + 1),
																		running_a2_col / (step + 1),
																		running_a3_col / (step + 1)))

			

		global_epoch += 1


def validate_model(val_data_loader, model):

	'''
	This function validates the model
	Args:
		- val_data_loader: the validation data loader
		- model: the model to validate
	'''

	eval_steps = len(val_data_loader)
	print('Evaluating for {} steps'.format(eval_steps))
	
	losses = []
	losses_row, losses_col = [], []
	a1s_row, a2s_row, a3s_row, a1s_col, a2s_col, a3s_col = [], [], [], [], [], []
	for step, batch_sample in tqdm(enumerate(val_data_loader)):

		model.eval()

		video = batch_sample['video']  
		audio = batch_sample['audio']  

		# Extract log-mel filterbanks  
		mel, _, _, _ = wav2filterbanks(audio.cuda())
		
		# Get the temporal window of 1-second
		mel_chunk = torch.split(mel, split_size_or_sections=num_frames*4, dim=1)
		mel_chunk = torch.stack(mel_chunk, dim=1)  
		mel_chunk = mel_chunk.permute(0,1,3,2)[:, None]
		
		vid_chunk = torch.split(video, split_size_or_sections=num_frames, dim=2)
		vid_chunk = torch.stack(vid_chunk, dim=2)        
		
		B=vid_chunk.size(0)
		face_sequences = torch.cat([vid_chunk[:, :, i] for i in range(vid_chunk.size(2))], dim=0)
		audio_sequences = torch.cat([mel_chunk[:, :, i] for i in range(mel_chunk.size(2))], dim=0)
		
		with torch.cuda.amp.autocast():

			if args.use_dist_training:
				video_embedding, video_feat = model.module.forward_vid(face_sequences.cuda(), return_feats=True)
				video_embedding = torch.mean(video_embedding, axis=-1).unsqueeze(-1)
				audio_embedding = model.module.forward_aud(audio_sequences.cuda())
			else:
				video_embedding, video_feat = model.forward_vid(face_sequences.cuda(), return_feats=True)
				video_embedding = torch.mean(video_embedding, axis=-1).unsqueeze(-1)
				audio_embedding = model.forward_aud(audio_sequences.cuda())

			audio_embedding = torch.split(audio_embedding, B, dim=0) 						
			audio_embedding = torch.stack(audio_embedding, dim=2) 							
			audio_embedding = audio_embedding.squeeze(3) 									

			video_embedding = torch.split(video_embedding, B, dim=0) 					
			video_embedding = torch.stack(video_embedding, dim=2) 					
			video_embedding = video_embedding.squeeze(3)								

			loss_row, a1_row, a2_row, a3_row, loss_col, a1_col, a2_col, a3_col = get_sync_loss(audio_embedding, video_embedding)

		loss = loss_row + loss_col
		losses.append(loss.item())
		losses_row.append(loss_row.item())
		losses_col.append(loss_col.item())
		a1s_row.append(a1_row)
		a2s_row.append(a2_row)
		a3s_row.append(a3_row)
		a1s_col.append(a1_col)
		a2s_col.append(a2_col)
		a3s_col.append(a3_col)


	# Get the average validation scores 
	averaged_loss = sum(losses) / len(losses)
	averaged_loss_row = sum(losses_row) / len(losses_row)
	averaged_loss_col = sum(losses_col) / len(losses_col)
	a1_row = sum(a1s_row) / len(a1s_row)
	a2_row = sum(a2s_row) / len(a2s_row)
	a3_row = sum(a3s_row) / len(a3s_row)
	a1_col = sum(a1s_col) / len(a1s_col)
	a2_col = sum(a2s_col) / len(a2s_col)
	a3_col = sum(a3s_col) / len(a3s_col)
	print('Validation loss: {:.3f} | Loss-r: {:.3f}, Loss-c: {:.3f} | | A1r: {:.2f}, A2r: {:.2f}, A3r: {:.2f} | A1c: {:.2f}, A2c: {:.2f}, A3c: {:.2f}'
			.format(averaged_loss, averaged_loss_row, averaged_loss_col, a1_row, a2_row, a3_row, a1_col, a2_col, a3_col))

	return

def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

	'''
	This function saves the checkpoint
	Args:
		- model: the model to save
		- optimizer: the optimizer to save
		- step: the global step
		- checkpoint_dir: the directory to save the checkpoints
		- epoch: the global epoch
	'''

	checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
	optimizer_state = optimizer.state_dict() 

	if args.use_dist_training:
		save_on_master({
			"state_dict": model.state_dict(),
			"optimizer": optimizer_state,
			"global_step": step,
			"global_epoch": epoch,
		}, checkpoint_path)
	else:
		torch.save({
			"state_dict": model.state_dict(),
			"optimizer": optimizer_state,
			"global_step": step,
			"global_epoch": epoch,
		}, checkpoint_path)
	
	print("Saved checkpoint:", checkpoint_path)


def load_checkpoint(path, model, optimizer, reset_optimizer=False):

	'''
	This function loads the checkpoint and the optimizer if needed (to resume training)
	Args:
		- path: the path to the checkpoint
		- model: the model to load
		- optimizer: the optimizer to load
		- reset_optimizer: whether to reset the optimizer 
	Returns:
		- model: the model loaded
	'''

	global global_step
	global global_epoch

	if use_cuda:
		checkpoint = torch.load(path)
	else:
		checkpoint = torch.load(path, map_location="cpu")

	s = checkpoint["state_dict"]
	new_s = {}
	
	for k, v in s.items():
		if args.use_dist_training:
			if not k.startswith('module.'):
				new_s['module.'+k] = v
			else:
				new_s[k] = v
		else:
			new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	if not reset_optimizer:
		optimizer_state = checkpoint["optimizer"]
		if optimizer_state is not None:
			print("Load optimizer state from {}".format(path))
			optimizer.load_state_dict(checkpoint["optimizer"])
	global_step = checkpoint["global_step"]
	global_epoch = checkpoint["global_epoch"]
	print("Load checkpoint from: {}".format(path))

	return model


def initialize_model(args, model):

	'''
	This function initializes the model for distributed training
	Args:
		- args: the arguments given in the config file
		- model: the model to initialize
	Returns:
		- model: the initialized model
	'''

	# For multiprocessing distributed, DistributedDataParallel constructor
	# should always set the single device scope, otherwise,
	# DistributedDataParallel will use all available devices.
	print ("Using distributed data parallel with {} GPUs".format(torch.cuda.device_count()))
	if args.sync_bn:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

	if args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model.cuda(args.gpu)
		model = torch.nn.parallel.DistributedDataParallel(model, \
								device_ids=[args.local_rank], output_device=args.local_rank, \
								broadcast_buffers=False, find_unused_parameters=True)
		
	else:
		model.cuda()
		model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
			
	return model

def load_files(split, path):

	'''
	This function loads the train/validation files
	Args:
		- split: the split to load (train/val)
		- path: the path to the files needed for training/validation
	Returns:
		- filelist: the list of filepaths needed to load the data and train the model
	'''
	
	filelist = glob(os.path.join(path, split, "*/*.avi"))

	return filelist


if __name__ == "__main__":

	'''
	This is the main function to train the model
	'''

	# Load all the configurations from the config file
	args = load_args()
	num_frames = args.num_frames

	# Create directory to save the model checkpoints
	if not os.path.exists(args.checkpoint_dir): 
		os.makedirs(args.checkpoint_dir)


	# Load the train and val files
	filelist_train = load_files("train", args.data_path_videos)
	filelist_val = load_files("val", args.data_path_videos)
	print("Total train files: ", len(filelist_train))
	print("Total validation files: ", len(filelist_val))

	if len(filelist_train) == 0:
		print("No train files found! Please check the data path.")
		exit(0)
	if len(filelist_val) == 0:
		print("No validation files found! Please check the data path.")
		exit(0)

	# Set the keypoint train and val paths
	data_path_kps_train = os.path.join(args.data_path_kps, "train")
	data_path_kps_val = os.path.join(args.data_path_kps, "val")


	# Setup train and val datasets
	train_dataset = DataGenerator_RGB(filelist_train, data_path_kps_train, args.total_frames, args.num_frames, height=args.height, width=args.width, fps=args.fps, sample_rate=args.sample_rate)
	val_dataset = DataGenerator_RGB(filelist_val, data_path_kps_val, args.total_frames, args.num_frames, height=args.height, width=args.width, fps=args.fps, sample_rate=args.sample_rate)


	# Define model
	model = Transformer_RGB()
	
	# Initialize distributed training if needed
	if args.use_dist_training:
		init_distributed_mode(args)

		# Create distributed data loaders	
		train_sampler = DistributedSampler(train_dataset, shuffle=True)
		train_data_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, \
													sampler=train_sampler, num_workers=args.num_workers, collate_fn=lambda x: collate(x))
		val_data_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, \
												collate_fn=lambda x: collate(x))

		# Initialize model for distributed training
		model = initialize_model(args, model)
		torch.backends.cudnn.benchmark = True

	else:
		args.batch_size = args.n_gpu * args.batch_size  
		
		# Create data loaders
		train_data_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, \
													num_workers=args.num_workers, collate_fn=lambda x: collate(x))
		val_data_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, \
													collate_fn=lambda x: collate(x))
		
		# Initialize model for data-parallel or single GPU training
		if args.n_gpu > 1:
			print("Using data parallel with {} GPUs".format(args.n_gpu))
			model = nn.DataParallel(model)
		else:
			print("Using single GPU to train the model")
		model = model.cuda()


	total_batch = len(train_data_loader)
	print("Total train batch: ", total_batch)
	print('Total trainable params {:.3f}M'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000))

	# Setup the optimizer
	optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

	# Loss functions
	logloss = nn.CrossEntropyLoss()
	cos = nn.CosineSimilarity(dim=1)

	# Load the checkpoint to resume training 
	if args.checkpoint_path is not None:
		load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=False)

	# Start the training
	train(model, train_data_loader, val_data_loader, optimizer, \
			checkpoint_dir=args.checkpoint_dir, checkpoint_interval=args.ckpt_interval, \
			validation_interval=args.val_interval, nepochs=args.num_epochs)
