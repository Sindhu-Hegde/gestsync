#!/usr/bin/python

import sys, os, argparse, pickle, subprocess, cv2, math
import numpy as np
from shutil import rmtree, copy, copytree
from tqdm import tqdm

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy import signal

from ultralytics import YOLO

from decord import VideoReader

parser = argparse.ArgumentParser(description="FaceTracker")
parser.add_argument('--data_root', type=str, required=True, help='Path of the folder containing full uncropped videos')
parser.add_argument('--preprocessed_root', type=str, required=True, help='Path to save the output crops')
parser.add_argument('--temp_dir', type=str, required=True, help='Path to save intermediate results')
parser.add_argument('--metadata_root', type=str, required=True, help='Path to save metadata files')

parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--nshard', type=int, default=1)
parser.add_argument('--crop_scale', type=float, default=0, help='Scale bounding box')
parser.add_argument('--min_track', type=int, default=50, help='Minimum facetrack duration')
parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate')
parser.add_argument('--num_failed_det', type=int, default=25, help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--min_frame_size', type=int, default=64, help='Minimum frame size in pixels')
opt = parser.parse_args()


def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxB[3], boxB[3])

	interArea = max(0, xB - xA) * max(0, yB - yA)

	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

def track_shot(opt, scenefaces):
	iouThres = 0.5  # Minimum IOU between consecutive face detections
	tracks = []

	while True:
		track = []
		for framefaces in scenefaces:
			for face in framefaces:
				if track == []:
					track.append(face)
					framefaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						framefaces.remove(face)
						continue
				else:
					break

		if track == []:
			break
		elif len(track) > opt.min_track:
			framenum = np.array([f['frame'] for f in track])
			bboxes = np.array([np.array(f['bbox']) for f in track])

			frame_i = np.arange(framenum[0], framenum[-1] + 1)

			bboxes_i = []
			for ij in range(0, 4):
				interpfn = interp1d(framenum, bboxes[:, ij])
				bboxes_i.append(interpfn(frame_i))
			bboxes_i = np.stack(bboxes_i, axis=1)

			if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_frame_size:
				tracks.append({'frame': frame_i, 'bbox': bboxes_i})

	return tracks

def check_folder(folder):
	if os.path.exists(folder):
		return True
	return False

def del_folder(folder):
	if os.path.exists(folder):
		rmtree(folder)

def read_video(o, start_idx):
	with open(o, 'rb') as o:
		video_stream = VideoReader(o)
		if start_idx > 0:
			video_stream.skip_frames(start_idx)
		return video_stream

def crop_video(opt, track, cropfile, tight_scale=1):
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	vOut = cv2.VideoWriter(cropfile + '.avi', fourcc, opt.frame_rate, (480, 270))

	dets = {'x': [], 'y': [], 's': []}

	for det in track['bbox']:
		# Reduce the size of the bounding box by a small factor for tighter crops
		width = (det[2] - det[0]) * tight_scale
		height = (det[3] - det[1]) * tight_scale
		center_x = (det[0] + det[2]) / 2
		center_y = (det[1] + det[3]) / 2

		dets['s'].append(max(height, width) / 2)
		dets['y'].append(center_y)  # crop center y
		dets['x'].append(center_x)  # crop center x

	# Smooth detections
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

	videofile = os.path.join(opt.avi_dir, 'video.avi')
	frame_no_to_start = track['frame'][0]
	video_stream = cv2.VideoCapture(videofile)
	video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no_to_start)
	for fidx, frame in enumerate(track['frame']):
		cs = opt.crop_scale
		bs = dets['s'][fidx]  # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

		image = video_stream.read()[1]
		frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))

		my = dets['y'][fidx] + bsi  # BBox center Y
		mx = dets['x'][fidx] + bsi  # BBox center X

		face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
		vOut.write(cv2.resize(face, (480, 270)))
	video_stream.release()
	audiotmp = os.path.join(opt.tmp_dir, 'audio.wav')
	audiostart = (track['frame'][0]) / opt.frame_rate
	audioend = (track['frame'][-1] + 1) / opt.frame_rate

	vOut.release()

	# ========== CROP AUDIO FILE ==========

	command = ("ffmpeg -hide_banner -loglevel panic -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.avi_dir, 'audio.wav'), audiostart, audioend, audiotmp))
	output = subprocess.call(command, shell=True, stdout=None)

	copy(audiotmp, cropfile + '.wav')

	print('Mean pos: x %.2f y %.2f s %.2f' % (np.mean(dets['x']), np.mean(dets['y']), np.mean(dets['s'])))

	return {'track': track, 'proc_track': dets}

def inference_video(opt, padding=0):
	videofile = os.path.join(opt.avi_dir, 'video.avi')
	vidObj = cv2.VideoCapture(videofile)
	yolo_model = YOLO("yolov9c.pt")

	dets = []
	fidx = 0
	while True:
		success, image = vidObj.read()
		if not success:
			break

		image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Perform person detection
		results = yolo_model(image_np, verbose=False)
		detections = results[0].boxes

		dets.append([])
		for i, det in enumerate(detections):
			x1, y1, x2, y2 = det.xyxy[0]
			cls = det.cls[0]
			conf = det.conf[0]  
			if int(cls) == 0 and conf>0.7:  # Class 0 is 'person' in COCO dataset
				x1 = max(0, int(x1) - padding)
				y1 = max(0, int(y1) - padding)
				x2 = min(image_np.shape[1], int(x2) + padding)
				y2 = min(image_np.shape[0], int(y2) + padding)
				dets[-1].append({'frame': fidx, 'bbox': [x1, y1, x2, y2], 'conf': conf})

		fidx += 1

	savepath = os.path.join(opt.work_dir, 'person.pckl')

	with open(savepath, 'wb') as fil:
		pickle.dump(dets, fil)

	return dets

def scene_detect(opt):
	video_manager = VideoManager([os.path.join(opt.avi_dir, 'video.avi')])
	stats_manager = StatsManager()
	scene_manager = SceneManager(stats_manager)
	scene_manager.add_detector(ContentDetector())
	base_timecode = video_manager.get_base_timecode()

	video_manager.set_downscale_factor()
	video_manager.start()
	scene_manager.detect_scenes(frame_source=video_manager)
	scene_list = scene_manager.get_scene_list(base_timecode)

	savepath = os.path.join(opt.work_dir, 'scene.pckl')

	if scene_list == []:
		scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]

	with open(savepath, 'wb') as fil:
		pickle.dump(scene_list, fil)

	print('%s - scenes detected %d' % (os.path.join(opt.avi_dir, 'video.avi'), len(scene_list)))

	return scene_list

def process_video(file):

	print("Processing video: ", file)
	video_file_name = os.path.basename(file.strip())
	sd_dest_folder = os.path.join(opt.preprocessed_root, video_file_name[:-4])
	work_dest_folder = os.path.join(opt.metadata_root, video_file_name[:-4])
	c1 = check_folder(sd_dest_folder)
	c2 = check_folder(work_dest_folder)
	if all([c1, c2]):
		print("Video already processed: ", file)
		return

	del_folder(sd_dest_folder)
	del_folder(work_dest_folder)

	file = os.path.join(opt.data_root, video_file_name)
	setattr(opt, 'videofile', file)

	if os.path.exists(opt.work_dir):
		rmtree(opt.work_dir)

	if os.path.exists(opt.crop_dir):
		rmtree(opt.crop_dir)

	if os.path.exists(opt.avi_dir):
		rmtree(opt.avi_dir)

	if os.path.exists(opt.frames_dir):
		rmtree(opt.frames_dir)

	if os.path.exists(opt.tmp_dir):
		rmtree(opt.tmp_dir)

	os.makedirs(opt.work_dir)
	os.makedirs(opt.crop_dir)
	os.makedirs(opt.avi_dir)
	os.makedirs(opt.frames_dir)
	os.makedirs(opt.tmp_dir)

	command = ("ffmpeg -hide_banner -loglevel panic -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (opt.videofile, 
																os.path.join(opt.avi_dir, 
																'video.avi')))
	output = subprocess.call(command, shell=True, stdout=None)
	if output != 0:
		return

	command = ("ffmpeg -hide_banner -loglevel panic -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,
																			 'video.avi'), 
																			 os.path.join(opt.avi_dir, 
																			'audio.wav')))
	output = subprocess.call(command, shell=True, stdout=None)
	if output != 0:
		return

	faces = inference_video(opt)

	try:
		scene = scene_detect(opt)
	except scenedetect.video_stream.VideoOpenFailure:
		return

	alltracks = []
	vidtracks = []

	for shot in scene:
		if shot[1].frame_num - shot[0].frame_num >= opt.min_track:
			alltracks.extend(track_shot(opt, faces[shot[0].frame_num:shot[1].frame_num]))

	for ii, track in enumerate(alltracks):
		vidtracks.append(crop_video(opt, track, os.path.join(opt.crop_dir, '%05d' % ii)))

	savepath = os.path.join(opt.work_dir, 'tracks.pckl')

	with open(savepath, 'wb') as fil:
		pickle.dump(vidtracks, fil)

	rmtree(opt.tmp_dir)
	rmtree(opt.avi_dir)
	rmtree(opt.frames_dir)
	copytree(opt.crop_dir, sd_dest_folder)
	copytree(opt.work_dir, work_dest_folder)


if __name__ == "__main__":

	files = os.listdir(opt.data_root)
	print(f"A total of {len(files)} files found.")

	os.makedirs(opt.preprocessed_root, exist_ok=True)
	os.makedirs(opt.metadata_root, exist_ok=True)

	num_per_shard = math.ceil(len(files) / opt.nshard)
	files = files[int(opt.rank * num_per_shard): int((opt.rank + 1) * num_per_shard)]
	print("Processing ", len(files), "videos on rank ", opt.rank)

	setattr(opt, 'avi_dir', os.path.join(opt.temp_dir, 'pyavi'))
	setattr(opt, 'tmp_dir', os.path.join(opt.temp_dir, 'pytmp'))
	setattr(opt, 'work_dir', os.path.join(opt.temp_dir, 'pywork'))
	setattr(opt, 'crop_dir', os.path.join(opt.temp_dir, 'pycrop'))
	setattr(opt, 'frames_dir', os.path.join(opt.temp_dir, 'pyframes'))


	for file in tqdm(files):
		process_video(file)

