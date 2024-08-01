import os
import pickle5 as pickle
import math
import numpy as np
import cv2
import random
from glob import glob

import torch
from torch.utils import data
from utils.audio_utils import *

from decord import VideoReader
from decord import cpu, gpu

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class DataGenerator_RGB(data.Dataset):

	def __init__(self, filelist, data_path_kps, total_frames, num_frames, height=270, width=480, fps=25, sample_rate=16000):

		self.files = filelist
		self.data_path_kps = data_path_kps
		self.total_frames = total_frames
		self.num_frames = num_frames
		self.height = height
		self.width = width
		self.fps = fps
		self.sample_rate = sample_rate
		self.face_oval_idx = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 
								176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]
	

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):

		# Get the name of the file
		fname = self.files[index]

		# Load the frames
		video, start_frame = self.load_frames(fname)

		if video is None:
			return None
		
		# Load the corresponding audio
		audio, aud_fact = self.load_audio(fname, start_frame)
		if audio is None:
			return None
					
		if not aud_fact * video.shape[1] == audio.shape[0]:
			return None
	
		# Convert to Torch tensor
		video = torch.FloatTensor(np.array(video))
		audio = torch.FloatTensor(audio)

		out_dict = {
			'video': video,
			'audio': audio,
			'file': fname
		}

		return out_dict

	def load_frames(self, fname):

		try:
			# Read the video
			vr = VideoReader(fname, ctx=cpu(0))

			kp_fname = os.path.join(self.data_path_kps, fname.split("/")[-2], fname.split("/")[-1].split(".")[0]+"_mediapipe_kps.pkl")
			kp_file = open(kp_fname, 'rb')
			kp_dict = pickle.load(kp_file)
		
			keypoints, resolution = kp_dict['kps'], kp_dict['resolution']

			# Randomly select the start frame		
			start_frame = random.choice(range(0, len(vr)-self.total_frames-1))
			end_frame = start_frame + self.total_frames

			# Give a 1-second gap between each window
			selected_frames = []
			selected_keypoints = []
			for i in range(start_frame, end_frame, self.num_frames*2):
				start, end = i, i+self.num_frames
				selected = range(start, end)
				frames_batch = vr.get_batch(selected).asnumpy()
				selected_frames.extend(frames_batch)
				selected_keypoints.extend(keypoints[start:end])
		

			# Mask the frames (since we do not use face information)
			input_frames = np.array(selected_frames)
			input_frames_masked = []
		
			for i, frame_kp_dict in enumerate(selected_keypoints):

				img = input_frames[i]

				face = frame_kp_dict["face"]

				face_kps = []
				for idx in range(len(face)):
					if idx in self.face_oval_idx:
						x, y = int(face[idx]["x"]*resolution[1]), int(face[idx]["y"]*resolution[0])
						face_kps.append((x,y))

				face_kps = np.array(face_kps)
				x1, y1 = min(face_kps[:,0]), min(face_kps[:,1])
				x2, y2 = max(face_kps[:,0]), max(face_kps[:,1])
				masked_img = cv2.rectangle(img, (0,0), (resolution[1],y2+15), (0,0,0), -1)
				
				if masked_img.shape[0] != self.height or masked_img.shape[1] != self.width:
					masked_img = cv2.resize(masked_img, (self.width, self.height))

				input_frames_masked.append(masked_img)
		except:
			return None, None


		input_frames = np.asarray(input_frames_masked) / 255.		# num_framesx270x480x3 
		input_frames = np.transpose(input_frames, (3, 0, 1, 2))		# 3xnum_framesx270x480
		
		return input_frames, start_frame

		
	def load_audio(self, fname, start_frame):
			
		# Load the audio file
		try:
			wavpath = fname.replace("vid", "wav")
			wavpath = wavpath.replace(".avi", ".wav")
			audio = load_wav(wavpath).astype('float32')
		except:
			return None, None

		# Load the corresponding audio window based on the selected start and end frame
		aud_fact = int(np.round(self.sample_rate / self.fps))

		audio_window = []
		end_frame = start_frame + self.total_frames
		for i in range(start_frame, end_frame, self.num_frames*2):
			start, end = aud_fact*i, aud_fact*(i+self.num_frames)
			audio_window.extend(audio[start:end])

		audio_window = np.array(audio_window)

		return audio_window, aud_fact

def collate(data):

	video = []
	audio = []
	files = []

	for sample in data:

		if sample is None:
			continue

		vid = sample['video']
		aud = sample['audio']
		file = sample['file']

		video.append(vid)
		audio.append(aud)
		files.append(file)

	if len(video)>0:
		video = torch.stack(video)
		audio = torch.stack(audio)
	else:
		return 0

	out_dict = {
			'video': video,
			'audio': audio,
			'file': files,
		}

	return out_dict	
