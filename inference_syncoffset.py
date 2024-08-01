import argparse
import os, subprocess

import numpy as np
import cv2

import librosa

import torch
# from torch.nn import functional as F

from utils.audio_utils import *
from utils.inference_utils import *
from sync_models.gestsync_models import *

from tqdm import tqdm
from scipy.io.wavfile import write

import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict
mp_holistic = mp.solutions.holistic
# import yolov5
from ultralytics import YOLO

# import decord
from decord import VideoReader, cpu

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


parser = argparse.ArgumentParser(description='Inference code for GestSync to sync-correct the video')

parser.add_argument('--checkpoint_path', required=True, help='Path of the trained model', default=None, type=str)
parser.add_argument('--video_path', default=None, required=True)
parser.add_argument('--num_avg_frames', type=int, default=50, choices=range(25, 1000))
parser.add_argument('--use_rgb', default=True)
parser.add_argument('--result_path', default="results")

parser.add_argument('--height', type=int, default=270)
parser.add_argument('--width', type=int, default=480)
parser.add_argument('--fps', type=int, default=25)
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--n_negative_samples', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=12)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

def preprocess_video(path, result_folder, padding=20):

	'''
	This function preprocesses the input video to extract the audio and crop the frames using YOLO model

	Args:
		- path (string) : Path of the input video file
		- result_folder (string) : Path of the folder to save the extracted audio and cropped video
		- padding (int) : Padding to add to the bounding box
	Returns:
		- wav_file (string) : Path of the extracted audio file
		- fps (int) : FPS of the input video
		- video_output (string) : Path of the cropped video file
	'''
	
	# Load all video frames
	vr = VideoReader(path, ctx=cpu(0))
	fps = vr.get_avg_fps()
	frame_count = len(vr)

	all_frames = []
	for k in range(len(vr)):
		all_frames.append(vr[k].asnumpy())
	all_frames = np.asarray(all_frames)

	# Load YOLOv5 model (pre-trained on COCO dataset)
	yolo_model = YOLO("yolov9c.pt")


	if frame_count < 25:
		print("Not enough frames to process! Please give a longer video as input")
		exit(0)

	person_videos = {}
	person_tracks = {}

	for frame_idx in range(frame_count):
	
		frame = all_frames[frame_idx]
	
		# Perform person detection
		results = yolo_model(frame, verbose=False)
		detections = results[0].boxes
	
		for i, det in enumerate(detections):
			x1, y1, x2, y2 = det.xyxy[0]
			cls = det.cls[0]
			if int(cls) == 0:  # Class 0 is 'person' in COCO dataset
			
				x1 = max(0, int(x1) - padding)
				y1 = max(0, int(y1) - padding)
				x2 = min(frame.shape[1], int(x2) + padding)
				y2 = min(frame.shape[0], int(y2) + padding)

				if i not in person_videos:
					person_videos[i] = []
					person_tracks[i] = []

				person_videos[i].append(frame)
				person_tracks[i].append([x1,y1,x2,y2])
		
	
	num_persons = 0
	for i in person_videos.keys():
		if len(person_videos[i]) >= frame_count//2:
			num_persons+=1

	if num_persons==0:
		print("No person detected in the video! Please give a video with one person as input")
		exit(0)
	if num_persons>1:
		print("More than one person detected in the video! Please give a video with only one person as input")
		exit(0)

	# Extract the audio from the input video file using ffmpeg
	try:
		wav_file  = os.path.join(result_folder, "audio.wav")

		subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -async 1 -ac 1 -vn \
						-acodec pcm_s16le -ar 16000 %s -y' % (path, wav_file), shell=True)

	except:
		print("Oops! Could not load the audio file. Please check the input video and try again.")
		exit(0)

	# For the person detected, crop the frame based on the bounding box
	if len(person_videos[0]) > frame_count-10:
		crop_filename = os.path.join(result_folder, "preprocessed_video.avi")
		fourcc = cv2.VideoWriter_fourcc(*'DIVX')

		# Get bounding box coordinates based on person_tracks[i]
		max_x1 = min([track[0] for track in person_tracks[0]])
		max_y1 = min([track[1] for track in person_tracks[0]])
		max_x2 = max([track[2] for track in person_tracks[0]])
		max_y2 = max([track[3] for track in person_tracks[0]])

		max_width = max_x2 - max_x1
		max_height = max_y2 - max_y1

		out = cv2.VideoWriter(crop_filename, fourcc, fps, (max_width, max_height))
		for frame in person_videos[0]:
			crop = frame[max_y1:max_y2, max_x1:max_x2]
			crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
			out.write(crop)
		out.release()

		no_sound_video = crop_filename.split('.')[0] + '_nosound.mp4'
		subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -c copy -an -strict -2 %s' % (crop_filename, no_sound_video), shell=True)
		
		video_output = crop_filename.split('.')[0] + '.mp4'
		subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 %s' % 
						(wav_file , no_sound_video, video_output), shell=True)
		
		os.remove(crop_filename)
		os.remove(no_sound_video)

		print("Successfully saved the pre-processed video: ", video_output)
	else:
		print("Could not track the person in the full video! Please give a single-speaker video as input")
		exit(0)

	return wav_file, fps, video_output


def load_checkpoint(path, model):
	'''
	This function loads the trained model from the checkpoint

	Args:
		- path (string) : Path of the checkpoint file
		- model (object) : Model object
	Returns:
		- model (object) : Model object with the weights loaded from the checkpoint
	'''	

	# Load the checkpoint
	if use_cuda:
		checkpoint = torch.load(path)
	else:
		checkpoint = torch.load(path, map_location="cpu")
	
	s = checkpoint["state_dict"]
	new_s = {}
	
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)
	model.cuda()

	print("Loaded checkpoint from: {}".format(path))

	return model.eval()


def load_video_frames(video_file):
	'''
	This function extracts the frames from the video

	Args:
		- video_file (string) : Path of the video file
	Returns:
		- frames (list) : List of frames extracted from the video
	'''

	# Read the video
	try:
		vr = VideoReader(video_file, ctx=cpu(0))
	except:
		print("Oops! Could not load the input video file")
		exit(0)


	# Extract the frames
	frames = []
	for k in range(len(vr)):
		frames.append(vr[k].asnumpy())

	frames = np.asarray(frames)

	return frames



def get_keypoints(frames):

	'''
	This function extracts the keypoints from the frames using MediaPipe Holistic pipeline

	Args:
		- frames (list) : List of frames extracted from the video
	Returns:
		- kp_dict (dict) : Dictionary containing the keypoints and the resolution of the frames
	'''

	holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) 

	resolution = frames[0].shape
	all_frame_kps = []

	for frame in frames:

		results = holistic.process(frame)

		pose, left_hand, right_hand, face = None, None, None, None
		if results.pose_landmarks is not None:
			pose = protobuf_to_dict(results.pose_landmarks)['landmark']
		if results.left_hand_landmarks is not None:
			left_hand = protobuf_to_dict(results.left_hand_landmarks)['landmark']
		if results.right_hand_landmarks is not None:
			right_hand = protobuf_to_dict(results.right_hand_landmarks)['landmark']
		if results.face_landmarks is not None:
			face = protobuf_to_dict(results.face_landmarks)['landmark']

		frame_dict = {"pose":pose, "left_hand":left_hand, "right_hand":right_hand, "face":face}

		all_frame_kps.append(frame_dict)

	kp_dict = {"kps":all_frame_kps, "resolution":resolution}

	return kp_dict


def check_visible_gestures(kp_dict):

	keypoints = kp_dict['kps']
	keypoints = np.array(keypoints)

	if len(keypoints)<25:
		return None, None
	
	ignore_idx = [0,1,2,3,4,5,6,7,8,9,10]

	count=0
	frame_pose_kps = []
	for frame_kp_dict in keypoints:

		pose = frame_kp_dict["pose"]
		left_hand = frame_kp_dict["left_hand"]
		right_hand = frame_kp_dict["right_hand"]

		if pose is None:
			continue
		
		if left_hand is None and right_hand is None:
			count+=1

		pose_kps = []
		for idx in range(len(pose)):
			# Ignore face keypoints
			if idx not in ignore_idx:
				x, y = pose[idx]["x"], pose[idx]["y"]
				pose_kps.append((x,y))

		frame_pose_kps.append(pose_kps)


	if count/len(keypoints) > 0.7 or len(frame_pose_kps)/len(keypoints) < 0.3:
		print("The gestures in the input video are not visible! Please give a video with visible gestures as input.")
		exit(0)

	print("Successfully verified the input video - Gestures are visible!")

def load_rgb_masked_frames(input_frames, kp_dict, stride=1, window_frames=25, width=480, height=270):

	'''
	This function masks the faces using the keypoints extracted from the frames

	Args:
		- input_frames (list) : List of frames extracted from the video
		- kp_dict (dict) : Dictionary containing the keypoints and the resolution of the frames
		- stride (int) : Stride to extract the frames
		- window_frames (int) : Number of frames in each window that is given as input to the model
		- width (int) : Width of the frames
		- height (int) : Height of the frames
	Returns:
		- input_frames (array) : Frame window to be given as input to the model
		- num_frames (int) : Number of frames to extract
		- orig_masked_frames (array) : Masked frames extracted from the video
	'''

	# Face indices to extract the face-coordinates needed for masking
	face_oval_idx = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 
					176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]

	
	input_keypoints, resolution = kp_dict['kps'], kp_dict['resolution']

	input_frames_masked = []
	for i, frame_kp_dict in enumerate(input_keypoints):

		img = input_frames[i]
		face = frame_kp_dict["face"]

		if face is None:
			img = cv2.resize(img, (width, height))
			masked_img = cv2.rectangle(img, (0,0), (width,120), (0,0,0), -1)
		else:
			face_kps = []
			for idx in range(len(face)):
				if idx in face_oval_idx:
					x, y = int(face[idx]["x"]*resolution[1]), int(face[idx]["y"]*resolution[0])
					face_kps.append((x,y))

			face_kps = np.array(face_kps)
			x1, y1 = min(face_kps[:,0]), min(face_kps[:,1])
			x2, y2 = max(face_kps[:,0]), max(face_kps[:,1])
			masked_img = cv2.rectangle(img, (0,0), (resolution[1],y2+15), (0,0,0), -1)

		if masked_img.shape[0] != width or masked_img.shape[1] != height:
			masked_img = cv2.resize(masked_img, (width, height))

		input_frames_masked.append(masked_img)

	orig_masked_frames = np.array(input_frames_masked)
	input_frames = np.array(input_frames_masked) / 255.
	# print("Input images full: ", input_frames.shape)      	# num_framesx270x480x3 

	input_frames = np.array([input_frames[i:i+window_frames, :, :] for i in range(0,input_frames.shape[0], stride) if (i+window_frames <= input_frames.shape[0])])
	# print("Input images window: ", input_frames.shape)      	# Tx25x270x480x3
	
	num_frames = input_frames.shape[0]

	if num_frames<10:
		print("Not enough frames to process! Please give a longer video as input.")
		exit(0)
	
	return input_frames, num_frames, orig_masked_frames

def load_spectrograms(wav_file, num_frames, window_frames=25, stride=4):

	'''
	This function extracts the spectrogram from the audio file

	Args:
		- wav_file (string) : Path of the extracted audio file
		- num_frames (int) : Number of frames to extract
		- result_folder (string) : Path of the folder to save the extracted audio file
		- window_frames (int) : Number of frames in each window that is given as input to the model
		- stride (int) : Stride to extract the audio frames
	Returns:
		- spec (array) : Spectrogram array window to be used as input to the model
		- orig_spec (array) : Spectrogram array extracted from the audio file
	'''

	# Extract the audio from the input video file using ffmpeg
	try:
		wav = librosa.load(wav_file, sr=16000)[0]
	except:
		print("Oops! Could extract the spectrograms from the audio file. Please check the input and try again.")
		exit(0)
	
	# Convert to tensor
	wav = torch.FloatTensor(wav).unsqueeze(0)
	mel, _, _, _ = wav2filterbanks(wav.to(device))
	spec = mel.squeeze(0).cpu().numpy()
	orig_spec = spec
	spec = np.array([spec[i:i+(window_frames*stride), :] for i in range(0, spec.shape[0], stride) if (i+(window_frames*stride) <= spec.shape[0])])

	if len(spec) != num_frames:
		spec = spec[:num_frames]
		frame_diff = np.abs(len(spec) - num_frames)
		if frame_diff > 60:
			print("The input video and audio length do not match - The results can be unreliable! Please check the input video.")

	return spec, orig_spec


def calc_optimal_av_offset(vid_emb, aud_emb):
	'''
	This function calculates the audio-visual offset between the video and audio

	Args:
		- vid_emb (array) : Video embedding array
		- aud_emb (array) : Audio embedding array
	Returns:
		- offset (int) : Optimal audio-visual offset
	'''

	pos_vid_emb, all_aud_emb, pos_idx, stride = create_online_sync_negatives(vid_emb, aud_emb)
	scores, _ = calc_av_scores(pos_vid_emb, all_aud_emb)
	offset = scores.argmax()*stride - pos_idx

	return offset.item()

def create_online_sync_negatives(vid_emb, aud_emb, stride=5):

	'''
	This function creates all possible positive and negative audio embeddings to compare and obtain the sync offset

	Args:
		- vid_emb (array) : Video embedding array
		- aud_emb (array) : Audio embedding array
		- stride (int) : Stride to extract the negative windows
	Returns:
		- vid_emb_pos (array) : Positive video embedding array
		- aud_emb_posneg (array) : All possible combinations of audio embedding array 
		- pos_idx_frame (int) : Positive video embedding array frame
		- stride (int) : Stride used to extract the negative windows
	'''

	slice_size = args.num_avg_frames
	aud_emb_posneg = aud_emb.squeeze(1).unfold(-1, slice_size, stride)
	aud_emb_posneg = aud_emb_posneg.permute([0, 2, 1, 3])
	aud_emb_posneg = aud_emb_posneg[:, :int(args.n_negative_samples/stride)+1]
	# print("Aud emb posneg: ", aud_emb_posneg.shape)

	pos_idx = (aud_emb_posneg.shape[1]//2)
	pos_idx_frame = pos_idx*stride

	min_offset_frames = -(pos_idx)*stride
	max_offset_frames = (aud_emb_posneg.shape[1] - pos_idx - 1)*stride
	print("With the current video length and the number of average frames, the model can predict the offsets in the range: [{}, {}]".format(min_offset_frames, max_offset_frames))

	vid_emb_pos = vid_emb[:, :, pos_idx_frame:pos_idx_frame+slice_size]
	if vid_emb_pos.shape[2] != slice_size:
		print("Video is too short to use {} frames to average the scores. Please use a longer input video or reduce the number of average frames".format(slice_size))
		exit(0)
	
	return vid_emb_pos, aud_emb_posneg, pos_idx_frame, stride

def calc_av_scores(vid_emb, aud_emb):

	'''
	This function calls functions to calculate the audio-visual similarity and attention map between the video and audio embeddings

	Args:
		- vid_emb (array) : Video embedding array
		- aud_emb (array) : Audio embedding array
	Returns:
		- scores (array) : Audio-visual similarity scores
		- att_map (array) : Attention map
	'''

	scores = calc_att_map(vid_emb, aud_emb)
	att_map = logsoftmax_2d(scores)
	scores = scores.mean(-1)
	
	return scores, att_map

def calc_att_map(vid_emb, aud_emb):

	'''
	This function calculates the similarity between the video and audio embeddings

	Args:
		- vid_emb (array) : Video embedding array
		- aud_emb (array) : Audio embedding array
	Returns:
		- scores (array) : Audio-visual similarity scores
	'''

	vid_emb = vid_emb[:, :, None]
	aud_emb = aud_emb.transpose(1, 2)

	scores = run_func_in_parts(lambda x, y: (x * y).sum(1),
							   vid_emb,
							   aud_emb,
							   part_len=10,
							   dim=3,
							   device=device)

	scores = model.logits_scale(scores[..., None]).squeeze(-1)

	return scores

def generate_video(frames, audio_file, video_fname):
	
	'''
	This function generates the video from the frames and audio file

	Args:
		- frames (array) : Frames to be used to generate the video
		- audio_file (string) : Path of the audio file
		- video_fname (string) : Path of the video file
	'''	

	fname = 'inference.avi'
	video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), 25, (frames[0].shape[1], frames[0].shape[0]))

	for i in range(len(frames)):
		video.write(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
	video.release()
	
	no_sound_video = video_fname + '_nosound.mp4'
	subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -c copy -an -strict -2 %s' % (fname, no_sound_video), shell=True)
	
	video_output = video_fname + '.mp4'
	subprocess.call('ffmpeg -hide_banner -loglevel panic -y -i %s -i %s -strict -2 -q:v 1 -shortest %s' % 
					(audio_file, no_sound_video, video_output), shell=True)

	os.remove(fname)
	os.remove(no_sound_video)
	
	print("Successfully generated the video:", video_output)

def sync_correct_video(frames, wav_file, offset, result_folder, sample_rate=16000, fps=25):

	'''
	This function corrects the video and audio to sync with each other

	Args:
		- frames (array) : Frames to be used to generate the video
		- wav_file (string) : Path of the audio file
		- offset (int) : Predicted sync-offset to be used to correct the video
		- result_folder (string) : Path of the result folder to save the output sync-corrected video
		- sample_rate (int) : Sample rate of the audio
		- fps (int) : Frames per second of the video
	'''

	if offset == 0:
		print("The input audio and video are in-sync! No need to perform sync correction.")
		return
	
	print("Performing Sync Correction...")
	corrected_frames = np.zeros_like(frames)
	if offset > 0:
		# corrected_frames[offset:] = frames[0:len(frames)-offset]
		audio_offset = int(offset*(sample_rate/fps))
		wav = librosa.core.load(wav_file, sr=sample_rate)[0]
		corrected_wav = wav[audio_offset:]
		corrected_wav_file = os.path.join(result_folder, "audio_sync_corrected.wav")
		write(corrected_wav_file, sample_rate, corrected_wav)
		wav_file = corrected_wav_file
		corrected_frames = frames
	elif offset < 0:
		corrected_frames[0:len(frames)+offset] = frames[np.abs(offset):]
	# print("Corrected frames: ", corrected_frames.shape)

	corrected_video_path = os.path.join(result_folder, "result_sync_corrected")
	generate_video(corrected_frames, wav_file, corrected_video_path)

	


if __name__ == "__main__":

	# Set the video path
	vid_path_orig = args.video_path 
	
	# Create folders to save the inputs and results
	result_folder = os.path.join(args.result_path, vid_path_orig.split("/")[-1].split(".")[0])
	if not os.path.exists(result_folder): 
		os.makedirs(result_folder)

	result_folder_input = os.path.join(result_folder, "input")
	if not os.path.exists(result_folder_input): 
		os.makedirs(result_folder_input)

	result_folder_output = os.path.join(result_folder, "output")
	if not os.path.exists(result_folder_output): 
		os.makedirs(result_folder_output)

	# Copy the input video to the result folder
	subprocess.call('rsync -az {} {}'.format(vid_path_orig, result_folder), shell=True)

	# Pre-process the input video
	wav_file, fps, vid_path_processed = preprocess_video(vid_path_orig, result_folder_input)
	print("FPS of video: ", fps)

	# Resample the video to 25 fps if it is not already 25 fps
	if fps!=25:
		vid_path = os.path.join(result_folder_input, "preprocessed_video_25fps.mp4")
		command = ("ffmpeg -hide_banner -loglevel panic -y -i {} -filter:v fps=25 {}".format(vid_path_processed, vid_path))
		from subprocess import call
		cmd = command.split(' ')
		print('Resampled the video to 25 fps: {}'.format(vid_path))
		call(cmd)
	else:
		vid_path = vid_path_processed

	# Load the video frames
	frames = load_video_frames(vid_path)
	orig_frames = frames.copy()
	
	# Check if the number of frames is enough to average the scores
	if len(frames) < args.num_avg_frames:
		print("The input video is too short to use {} frames to average the scores. Please use a longer input video or reduce the number of average frames".format(args.num_avg_frames))
		exit(0)

	# Load the keypoints
	kp_dict = get_keypoints(frames)

	# Check if the gestures are visible in the input video
	check_visible_gestures(kp_dict)

	# Load the trained model
	if args.use_rgb:		
		model = Transformer_RGB()
	else:
		print("Currently only RGB model is supported!")
		exit(0)
	model = load_checkpoint(args.checkpoint_path, model) 
	

	# Load the input frames (maksed frames for RGB model and keypoints for keypoint model)
	if args.use_rgb:
		print("Using RGB model")
		rgb_frames, num_frames, orig_masked_frames = load_rgb_masked_frames(frames, kp_dict, window_frames=25)
	else:
		print("Currently only RGB model is supported!")
		exit(0)

	
	# Convert the frames to tensor
	rgb_frames = np.transpose(rgb_frames, (4, 0, 1, 2, 3))
	rgb_frames = torch.FloatTensor(np.array(rgb_frames)).unsqueeze(0)
	B=rgb_frames.size(0)

	# Load the spectrograms
	spec, orig_spec = load_spectrograms(wav_file, num_frames, window_frames=25)
	spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).permute(0,1,2,4,3)

	# Save the masked video to the result folder
	input_masked_vid_path = os.path.join(result_folder_input, "input_masked_video")
	generate_video(orig_masked_frames, wav_file, input_masked_vid_path)

	# Create the input windows for the frames and spectrograms
	video_sequences = torch.cat([rgb_frames[:, :, i] for i in range(rgb_frames.size(2))], dim=0)
	audio_sequences = torch.cat([spec[:, :, i] for i in range(spec.size(2))], dim=0)

	batch_size = args.batch_size
	video_emb = []
	audio_emb = []
	video_feat = []

	# Obtain the video and audio embeddings
	for i in tqdm(range(0, len(video_sequences), batch_size)):
		video_inp = video_sequences[i:i+batch_size, ]
		audio_inp = audio_sequences[i:i+batch_size, ]
			
		vid_emb, vid_feat = model.forward_vid(video_inp.to(device), return_feats=True)
		vid_emb = torch.mean(vid_emb, axis=-1).unsqueeze(-1)
		aud_emb = model.forward_aud(audio_inp.to(device))

		video_emb.append(vid_emb.detach())
		video_feat.append(vid_feat.detach())
		audio_emb.append(aud_emb.detach())
		
		torch.cuda.empty_cache()
		
	audio_emb = torch.cat(audio_emb, dim=0)
	video_emb = torch.cat(video_emb, dim=0)
	video_feat = torch.cat(video_feat, dim=0)

	# print("Audio emb: ", audio_emb.shape)
	# print("Video emb: ", video_emb.shape)

	# L2 normalize the embeddings
	video_emb = torch.nn.functional.normalize(video_emb, p=2, dim=1)
	audio_emb = torch.nn.functional.normalize(audio_emb, p=2, dim=1)

	audio_emb = torch.split(audio_emb, B, dim=0)                        # Bx512x1
	audio_emb = torch.stack(audio_emb, dim=2)                           # Bx512xTx1
	audio_emb = audio_emb.squeeze(3)                                    # Bx512xT
	audio_emb = audio_emb[:, None]

	video_emb = torch.split(video_emb, B, dim=0)                        # Bx512x1
	video_emb = torch.stack(video_emb, dim=2)                           # Bx512xTx1
	video_emb = video_emb.squeeze(3)                                    # Bx512xT

	# Calculate the sync offset
	pred_offset = calc_optimal_av_offset(video_emb, audio_emb)
	print("Predicted offset: ", pred_offset)

	# Generate and save the sync-corrected video
	sync_correct_video(orig_frames, wav_file, pred_offset, result_folder_output)