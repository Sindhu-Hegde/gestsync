import argparse
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os, subprocess
import cv2, pickle
import librosa
from decord import VideoReader
from decord import cpu, gpu

from utils.audio_utils import *
from utils.inference_utils import *
from sync_models.gestsync_models import *

from tqdm import tqdm
from glob import glob

import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict
mp_holistic = mp.solutions.holistic

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


parser = argparse.ArgumentParser(description='Inference code for GestSync to predict the active speaker')

parser.add_argument('--checkpoint_path', required=True, help='Path of the trained model', type=str)
parser.add_argument('--video_path', required=True, help='Path of the input video', type=str)
parser.add_argument('--global_speaker', help='Whether to predict the global speaker for the entire video or not', default="True",  type=str)
parser.add_argument('--num_avg_frames', help='Number of frames to average', default=25, type=int)
parser.add_argument('--result_path', help='Path of the result folder', default="results", type=str)

parser.add_argument('--height', default=270, type=int)
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--fps', default=25, type=int)
parser.add_argument('--sample_rate', default=16000, type=int)
parser.add_argument('--window_frames', default=25, type=int)
parser.add_argument('--batch_size', default=12, type=int)


args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# Initialize the mediapipe holistic keypoint detection model
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) 


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
			masked_img = cv2.rectangle(img, (0,0), (width,110), (0,0,0), -1)
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
	input_frames = np.pad(input_frames, ((12, 12), (0,0), (0,0), (0,0)), 'edge')
	# print("Input images full: ", input_frames.shape)      	# num_framesx270x480x3 

	input_frames = np.array([input_frames[i:i+window_frames, :, :] for i in range(0,input_frames.shape[0], stride) if (i+window_frames <= input_frames.shape[0])])
	# print("Input images window: ", input_frames.shape)      	# Tx25x270x480x3
	
	num_frames = input_frames.shape[0]

	if num_frames<10:
		print("Not enough frames to process! Please give a longer video as input.")
		exit(0)
	
	return input_frames, num_frames, orig_masked_frames


def load_masked_input_frames(test_videos, spec, wav_file, scene_num, result_folder):

	'''
	This function loads the masked input frames from the video

	Args:
		- test_videos (list) : List of videos to be processed (speaker-specific tracks)
		- spec (array) : Spectrogram of the audio
		- wav_file (string) : Path of the audio file
		- scene_num (int) : Scene number to be used to save the input masked video
		- result_folder (string) : Path of the folder to save the input masked video
	Returns:
		- all_frames (list) : List of masked input frames window to be used as input to the model
		- all_orig_frames (list) : List of original masked input frames
	'''

	all_frames, all_orig_frames = [], []
	for video_num, video in enumerate(test_videos):

		# Load the video frames
		frames = load_video_frames(video)

		# Check if the number of frames is enough to average the scores
		if args.global_speaker=="False" and len(frames) < args.num_avg_frames:
			print("The input video is too short to use {} frames to average the scores. Please use a longer input video or reduce the number of average frames".format(args.num_avg_frames))
			exit(0)

		# Extract the keypoints from the frames
		kp_dict = get_keypoints(frames)

		# Mask the frames using the keypoints extracted from the frames and prepare the input to the model
		masked_frames, num_frames, orig_masked_frames = load_rgb_masked_frames(frames, kp_dict)
		input_masked_vid_path = os.path.join(result_folder, "input_masked_scene_{}_speaker_{}".format(scene_num, video_num))
		generate_video(orig_masked_frames, wav_file, input_masked_vid_path)

		# Check if the length of the input frames is equal to the length of the spectrogram
		if spec.shape[2]!=masked_frames.shape[0]:
			num_frames = spec.shape[2]
			masked_frames = masked_frames[:num_frames]
			orig_masked_frames = orig_masked_frames[:num_frames]
			frame_diff = np.abs(spec.shape[2] - num_frames)
			if frame_diff > 60:
				print("The input video and audio length do not match - The results can be unreliable! Please check the input video.")

		# Transpose the frames to the correct format
		frames = np.transpose(masked_frames, (4, 0, 1, 2, 3))
		frames = torch.FloatTensor(np.array(frames)).unsqueeze(0)

		all_frames.append(frames)
		all_orig_frames.append(orig_masked_frames)


	return all_frames, all_orig_frames

def extract_audio(video, result_folder):

	'''
	This function extracts the audio from the video file

	Args:
		- video (string) : Path of the video file
		- result_folder (string) : Path of the folder to save the extracted audio file
	Returns:
		- wav_file (string) : Path of the extracted audio file
	'''

	try:
		wav_file  = os.path.join(result_folder, "audio.wav")

		subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -async 1 -ac 1 -vn \
						-acodec pcm_s16le -ar 16000 %s' % (video, wav_file), shell=True)

	except:
		print("Oops! Could not load the audio file in the given input video. Please check the input and try again")
		exit(0)
	
	return wav_file


def load_spectrograms(wav_file, window_frames=25, stride=4):

	'''
	This function extracts the spectrogram from the audio file

	Args:
		- wav_file (string) : Path of the extracted audio file
		- window_frames (int) : Number of frames in each window that is given as input to the model
		- stride (int) : Stride to extract the audio frames
	Returns:
		- spec (array) : Spectrogram array window to be used as input to the model
		- orig_spec (array) : Spectrogram array extracted from the audio file
	'''


	try:
		wav = librosa.load(wav_file, sr=16000)[0]
		wav = torch.FloatTensor(wav).unsqueeze(0)
		mel, _, _, _ = wav2filterbanks(wav.to(device))
	except:
		print("Oops! Could extract the spectrograms from the audio file. Please check the input and try again.")
		exit(0)
	
	spec = mel.squeeze(0).cpu().numpy()
	orig_spec = spec
	spec = np.array([spec[i:i+(window_frames*stride), :] for i in range(0, spec.shape[0], stride) if (i+(window_frames*stride) <= spec.shape[0])])
	pad_frames = (window_frames//2)
	spec = np.pad(spec, ((pad_frames, pad_frames), (0,0), (0,0)), 'edge')

	return spec, orig_spec


def get_embeddings(video_sequences, audio_sequences, model, calc_aud_emb=True):

	'''
	This function extracts the video and audio embeddings from the input frames and audio sequences

	Args:
		- video_sequences (array) : Array of video frames to be used as input to the model
		- audio_sequences (array) : Array of audio frames to be used as input to the model
		- model (object) : Model object
		- calc_aud_emb (bool) : Flag to calculate the audio embedding
	Returns:
		- video_emb (array) : Video embedding
		- audio_emb (array) : Audio embedding
	'''

	batch_size = args.batch_size
	video_emb = []
	audio_emb = []

	for i in range(0, len(video_sequences), batch_size):
		video_inp = video_sequences[i:i+batch_size, ]  		
		vid_emb = model.forward_vid(video_inp.to(device), return_feats=False)
		vid_emb = torch.mean(vid_emb, axis=-1)

		video_emb.append(vid_emb.detach())

		if calc_aud_emb:
			audio_inp = audio_sequences[i:i+batch_size, ]
			aud_emb = model.forward_aud(audio_inp.to(device))
			audio_emb.append(aud_emb.detach())
		
		torch.cuda.empty_cache()

	video_emb = torch.cat(video_emb, dim=0)

	if calc_aud_emb:
		audio_emb = torch.cat(audio_emb, dim=0)

		return video_emb, audio_emb       

	return video_emb



def predict_active_speaker(all_video_embeddings, audio_embedding, global_score):
	
	'''
	This function predicts the active speaker in each frame

	Args:
		- all_video_embeddings (array) : Array of video embeddings of all speakers
		- audio_embedding (array) : Audio embedding
		- global_score (bool) : Flag to calculate the global score
	Returns:
		- pred_speaker (list) : List of active speakers in each frame
	'''

	cos = nn.CosineSimilarity(dim=1)

	audio_embedding = audio_embedding.squeeze(2)

	scores = []
	for i in range(len(all_video_embeddings)):
		video_embedding = all_video_embeddings[i]

		# Compute the similarity of each speaker's video embeddings with the audio embedding
		sim = cos(video_embedding, audio_embedding)

		# Apply the logits scale to the similarity scores (scaling the scores)
		output = model.logits_scale(sim.unsqueeze(-1)).squeeze(-1)

		if global_score=="True":
			score = output.mean(0)
		else:
			output_batch = output.unfold(0, args.num_avg_frames, 1)
			score = torch.mean(output_batch, axis=-1)

		scores.append(score.detach().cpu().numpy())

	if global_score=="True":
		pred_speaker = np.argmax(scores)
	else:
		pred_speaker = []
		num_negs = list(range(0, len(all_video_embeddings)))
		for frame_idx in range(len(scores[0])):
			score = [scores[i][frame_idx] for i in num_negs] 
			pred_idx = np.argmax(score)
			pred_speaker.append(pred_idx)

	return pred_speaker


def generate_video(frames, audio_file, video_fname, fps=25):
	
	'''
	This function generates the video from the frames and audio file

	Args:
		- frames (array) : Frames to be used to generate the video
		- audio_file (string) : Path of the audio file
		- video_fname (string) : Path of the video file
	Returns:
		- video_output (string) : Path of the output video
	'''	

	fname = 'output.avi'
	video = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frames[0].shape[1], frames[0].shape[0]))

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
	
	return video_output


def save_video(output_tracks, input_frames, wav_file, result_folder):

	'''
	This function saves the output video with the active speaker detections

	Args:
		- output_tracks (list) : List of active speakers in each frame
		- input_frames (array) : Frames to be used to generate the video
		- wav_file (string) : Path of the audio file
		- result_folder (string) : Path of the result folder to save the output video
	Returns:
		- video_output (string) : Path of the output video
	'''

	output_frames = []
	for i in range(len(input_frames)):

		# If the active speaker is found, draw a bounding box around the active speaker
		if i in output_tracks:
			bbox = output_tracks[i]
			x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
			out = cv2.rectangle(input_frames[i].copy(), (x1, y1), (x2, y2), color=[0, 255, 0], thickness=3)
		else:
			out = input_frames[i]		

		output_frames.append(out)

	# Generate the output video
	output_video_fname = os.path.join(result_folder, "result_active_speaker_det")
	video_output = generate_video(output_frames, wav_file, output_video_fname)     

	return video_output

def resample_video(video_file, video_fname, result_folder):

	'''
	This function resamples the video to 25 fps

	Args:
		- video_file (string) : Path of the input video
		- video_fname (string) : Path of the input video
		- result_folder (string) : Path of the result folder to save the output video
	Returns:
		- video_output (string) : Path of the output video
	'''

	video_file_25fps = os.path.join(result_folder, '{}_25fps.mp4'.format(video_fname))
	
	# Resample the video to 25 fps
	command = ("ffmpeg -hide_banner -loglevel panic -y -i {} -q:v 1 -filter:v fps=25 {}".format(video_file, video_file_25fps))
	from subprocess import call
	cmd = command.split(' ')
	print('Resampled the video to 25 fps: {}'.format(video_file_25fps))
	call(cmd)

	return video_file_25fps



if __name__ == "__main__":


	# Read the video
	try:
		vr = VideoReader(args.video_path, ctx=cpu(0))
	except:
		print("Oops! Could not load the input video file")
		exit(0)

	# Extract the video filename
	video_fname = os.path.basename(args.video_path.split(".")[0])
	
	# Create folders to save the inputs and results
	result_folder = os.path.join(args.result_path, video_fname)
	if not os.path.exists(result_folder): 
		os.makedirs(result_folder)

	result_folder_input = os.path.join(result_folder, "input")
	if not os.path.exists(result_folder_input): 
		os.makedirs(result_folder_input)

	result_folder_output = os.path.join(result_folder, "output")
	if not os.path.exists(result_folder_output): 
		os.makedirs(result_folder_output)

	if args.global_speaker=="False" and args.num_avg_frames<25:
		print("Number of frames to average need to be set to a minimum of 25 frames. Atleast 1-second context is needed for the model. Please change the num_avg_frames and try again...")
		exit(0)

	# Get the FPS of the video
	fps = vr.get_avg_fps()
	print("FPS of video: ", fps)

	# Resample the video to 25 FPS if the original video is of a different frame-rate
	if fps!=25:
		test_video_25fps = resample_video(args.video_path, video_fname, result_folder_input)
	else:
		test_video_25fps = args.video_path

	# Load the video frames
	orig_frames = load_video_frames(test_video_25fps)

	# Copy the input video to the result folder
	subprocess.call('rsync -az {} {}'.format(args.video_path, result_folder), shell=True)

	# Extract and save the audio file
	orig_wav_file = extract_audio(args.video_path, result_folder)

	# Pre-process and extract per-speaker tracks in each scene 
	print("Pre-processing the input video...")
	subprocess.call("python preprocess/inference_preprocess.py --data_dir={}/temp --sd_root={}/crops --work_root={}/metadata --data_root={}".format(result_folder_input, result_folder_input, result_folder_input, args.video_path), shell=True)
	
	# Load the tracks file saved during pre-processing
	with open('{}/metadata/tracks.pckl'.format(result_folder_input), 'rb') as file:
		tracks = pickle.load(file)


	# Create a dictionary of all tracks found along with the bounding-boxes
	track_dict = {}
	for scene_num in range(len(tracks)):
		track_dict[scene_num] = {}
		for i in range(len(tracks[scene_num])):
			track_dict[scene_num][i] = {}
			for frame_num, bbox in zip(tracks[scene_num][i]['track']['frame'], tracks[scene_num][i]['track']['bbox']):
				track_dict[scene_num][i][frame_num] = bbox

	# Get the total number of scenes 
	test_scenes = os.listdir("{}/crops".format(result_folder_input))
	print("Total scenes found in the input video = ", len(test_scenes))

	# Load the trained model
	model = Transformer_RGB()
	model = load_checkpoint(args.checkpoint_path, model) 

	# Compute the active speaker in each scene 
	output_tracks = {}
	for scene_num in tqdm(range(len(test_scenes))):
		test_videos = glob(os.path.join("{}/crops".format(result_folder_input), "scene_{}".format(str(scene_num)), "*.avi"))
		test_videos.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
		print("Scene {} -> Total video files found (speaker-specific tracks) = {}".format(scene_num, len(test_videos)))

		if len(test_videos)<=1:
			print("To detect the active speaker, at least 2 visible speakers are required for each scene! Please check the input video and try again...")
			exit(0)

		# Load the audio file
		audio_file = glob(os.path.join("{}/crops".format(result_folder_input), "scene_{}".format(str(scene_num)), "*.wav"))[0]
		spec, _ = load_spectrograms(audio_file, window_frames=args.window_frames)
		spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).permute(0,1,2,4,3)
		
		# Load the masked input frames
		all_masked_frames, all_orig_masked_frames = load_masked_input_frames(test_videos, spec, audio_file, scene_num, result_folder_input)
		

		# Prepare the audio and video sequences for the model
		audio_sequences = torch.cat([spec[:, :, i] for i in range(spec.size(2))], dim=0)

		print("Obtaining audio and video embeddings...")
		all_video_embs = []
		for idx in tqdm(range(len(all_masked_frames))):
			with torch.no_grad():
				video_sequences = torch.cat([all_masked_frames[idx][:, :, i] for i in range(all_masked_frames[idx].size(2))], dim=0)

				if idx==0:
					video_emb, audio_emb = get_embeddings(video_sequences, audio_sequences, model, calc_aud_emb=True)
				else:
					video_emb = get_embeddings(video_sequences, audio_sequences, model, calc_aud_emb=False)
				all_video_embs.append(video_emb)

		# Predict the active speaker in each scene
		predictions = predict_active_speaker(all_video_embs, audio_emb, args.global_speaker)

		# Get the frames present in the scene
		frames_scene = tracks[scene_num][0]['track']['frame']

		# Prepare the active speakers list to draw the bounding boxes
		if args.global_speaker=="True":
			active_speakers = [predictions]*len(frames_scene)
			start, end = 0, len(frames_scene)
		else:
			active_speakers = [0]*len(frames_scene)
			mid = args.num_avg_frames//2

			if args.num_avg_frames%2==0:	
				frame_pred = len(frames_scene)-(mid*2)+1
				start, end = mid, len(frames_scene)-mid+1
			else:
				frame_pred = len(frames_scene)-(mid*2)
				start, end = mid, len(frames_scene)-mid

			if len(predictions) != frame_pred:
				print("Predicted frames {} and input video frames {} do not match!!".format(len(predictions), frame_pred))
				exit(0)

			active_speakers[start:end] = predictions[0:]

			# Depending on the num_avg_frames, interpolate the intial and final frame predictions to get a full video output
			initial_preds = max(set(predictions[:args.num_avg_frames]), key=predictions[:args.num_avg_frames].count)
			active_speakers[0:start] = [initial_preds] * start
			
			final_preds = max(set(predictions[-args.num_avg_frames:]), key=predictions[-args.num_avg_frames:].count)
			active_speakers[end:] = [final_preds] * (len(frames_scene) - end)
			start, end = 0, len(active_speakers)
	
		# Get the output tracks for each frame
		pred_idx = 0
		for frame in frames_scene[start:end]:
			label = active_speakers[pred_idx]
			pred_idx += 1
			output_tracks[frame] = track_dict[scene_num][label][frame]

	# Save the output video
	video_output = save_video(output_tracks, orig_frames.copy(), orig_wav_file, result_folder_output)
	print("Successfully saved the output video: ", video_output)
