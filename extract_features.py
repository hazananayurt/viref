import os
import csv
import pickle
import xml.etree.ElementTree as ET
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, model_from_json
import cv2
import numpy as np

	
	
def scale_bb(bb, frame_width, frame_height, to_size):
	x, y, width, height = bb
	return [int(x*to_size/frame_width), int(y*to_size/frame_height), int(width*to_size/frame_width), int(height*to_size/frame_height)]


def resize_frame(frame, final_resolution):
	if final_resolution is False:
		return frame
		
	desired_size = final_resolution
	old_size = frame.shape[:2]
	ratio = float(desired_size)/max(old_size)
	new_size = (int(ratio*old_size[0]), int(ratio*old_size[1]))

	frame = cv2.resize(frame, new_size[::-1], interpolation=cv2.INTER_NEAREST)

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]

	top = int(delta_h/2)
	bottom = delta_h - top
	left = int(delta_w/2)
	right = delta_w - left

	color = [0, 0, 0]
	new_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
	
	return new_frame
	

def blacken_frame(frame, bb_list):
	mask = np.zeros_like(frame, dtype=np.float32)
	for x, y, width, height in bb_list:
		mask[y:y+height, x:x+width, :] = 1.0
	return frame*mask


def register_to_path(video_name, register):
	return os.path.join(features_dir, register[0], ('{:s}_' + '_{:d}'*(len(register)-1)).format(video_name, *register[1:]))


def read_refexp(refexp_path):
	refexp_list = []
	with open(refexp_path, newline='') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			video_name, obj1_id, obj2_id, refexp = row
			obj1_id = int(obj1_id)
			obj2_id = int(obj2_id)
			refexp_list.append([video_name, obj1_id, obj2_id, refexp.split()])
	return refexp_list


def get_virat_bb_gt(video_name, obj_ids):
	sequences_dir = os.path.join(data_dir, 'VIRAT Ground Dataset', 'annotations')
	bb_gt = []
	with open(os.path.join(sequences_dir, '{:s}.viratdata.objects.txt'.format(video_name)), 'r') as f:
		for line in f:
			obj_id, _, frame_no, x, y , width, height, _ = line.split()
			obj_id, frame_no, x, y, width, height = int(obj_id), int(frame_no), int(x),\
													int(y), int(width), int(height)
			if obj_id not in obj_ids:
				continue
			bb_gt.append([obj_id, frame_no, x, y, width, height])
	return bb_gt


def get_ilsvrc_bb_gt(video_name, obj_ids):
	vid_train_dir = os.path.join(data_dir, 'ILSVRC2015', 'Annotations', 'VID', 'train')
	for	sequences_dir_name in os.listdir(vid_train_dir):
		sequences_dir = os.path.join(vid_train_dir, sequences_dir_name)
		if video_name in os.listdir(sequences_dir):
			sequence_dir = os.path.join(sequences_dir, video_name)
			break
	bb_gt = []
	for frame_filename in sorted(os.listdir(sequence_dir)):
		frame_no = int(frame_filename.split('.')[0])
		tree = ET.parse(os.path.join(sequence_dir, frame_filename))
		root = tree.getroot()
		for child in root:
			if child.tag == 'object':
				obj_id = int(child.find('trackid').text)
				if obj_id not in obj_ids:
					continue
				bb = child.find('bndbox')
				xmin = int(bb.find('xmin').text)
				xmax = int(bb.find('xmax').text)
				ymin = int(bb.find('ymin').text)
				ymax = int(bb.find('ymax').text)
				x = xmin
				y = ymin
				width = xmax - xmin
				height = ymax - ymin
				bb_gt.append([obj_id, frame_no, x, y, width, height])
	return bb_gt


def get_bb_gt(video_name, dataset_type, obj_ids):
	if dataset_type == 'virat':
		bb_gt = get_virat_bb_gt(video_name, obj_ids)
	elif dataset_type == 'ilsvrc':
		bb_gt = get_ilsvrc_bb_gt(video_name, obj_ids)
	return sorted(bb_gt, key=lambda row: row[1])


def get_minimum_and_maximum_frame_numbers(bb_gt):
	maximum_frame_number = 0
	minimum_frame_number = 10**10
	for bb in bb_gt:
		obj_id, frame_number, x, y, width, height = bb
		if frame_number > maximum_frame_number:
			maximum_frame_number = frame_number
		if frame_number < minimum_frame_number:
			minimum_frame_number = frame_number
	return minimum_frame_number, maximum_frame_number


def get_id_to_start_end(bb_gt, minimum_frame_number, maximum_frame_number):
	id_to_start_end = {}
	for bb in bb_gt:
		obj_id, frame_number, x, y, width, height = bb
		if obj_id not in id_to_start_end:
			id_to_start_end[obj_id] = [maximum_frame_number+1, -1] #start, end
		if frame_number < id_to_start_end[obj_id][0]:
			id_to_start_end[obj_id][0] = frame_number
		if frame_number > id_to_start_end[obj_id][1]:
			id_to_start_end[obj_id][1] = frame_number
	return id_to_start_end
	

def get_frame_to_bb(bb_gt, minimum_frame_number, maximum_frame_number):
	frame_to_bb = [{} for i in range(minimum_frame_number, maximum_frame_number+1)]
	for bb in bb_gt:
		obj_id, frame_number, x, y, width, height = bb
		frame_to_bb[frame_number-minimum_frame_number][obj_id] = (x, y, width, height)
	return frame_to_bb


def get_pair_start_end_list(id_to_start_end, sequences, minimum_frame_number, maximum_frame_number):
	pair_start_list = [[] for i in range(minimum_frame_number, maximum_frame_number+2)]
	pair_end_list = [[] for i in range(minimum_frame_number, maximum_frame_number+2)]
	for obj1_id, obj2_id in sequences:
		start = min(id_to_start_end[obj1_id][0], id_to_start_end[obj2_id][0])
		end = max(id_to_start_end[obj1_id][1], id_to_start_end[obj2_id][1])
		pair_start_list[start-minimum_frame_number].append((obj1_id, obj2_id))
		pair_end_list[end-minimum_frame_number+1].append((obj1_id, obj2_id))
	
	return pair_start_list, pair_end_list
	
def get_obj_start_end_list(id_to_start_end, minimum_frame_number, maximum_frame_number):
	obj_start_list = [[] for i in range(minimum_frame_number, maximum_frame_number+2)]
	obj_end_list = [[] for i in range(minimum_frame_number, maximum_frame_number+2)]
	for obj_id in id_to_start_end:
		start = id_to_start_end[obj_id][0]
		end = id_to_start_end[obj_id][1]
		obj_start_list[start-minimum_frame_number].append(obj_id)
		obj_end_list[end-minimum_frame_number+1].append(obj_id)
	
	return obj_start_list, obj_end_list
		
def get_video_feature(video_name, dataset_type):
	if dataset_type == "virat":
		videos_original_dir = os.path.join(data_dir, "VIRAT Ground Dataset", "videos_original")
		video_path = os.path.join(videos_original_dir, video_name+".mp4")
		cap = cv2.VideoCapture(video_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		ret, frame = cap.read()
		height, width = frame.shape[:2]
		cap.release()
	elif dataset_type == "ilsvrc":
		vid_train_dir = os.path.join(data_dir, 'ILSVRC2015', 'Data', 'VID', 'train')
		for	sequences_dir_name in os.listdir(vid_train_dir):
			sequences_dir = os.path.join(vid_train_dir, sequences_dir_name)
			if video_name in os.listdir(sequences_dir):
				frames_path = os.path.join(sequences_dir, video_name)
				break
		fps = 20
		frame = cv2.imread(os.path.join(frames_path, "{:06d}.JPEG".format(0)))
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
		height, width = frame.shape[:2]
	return fps, width, height


def frame_generator(video_name, dataset_type):
	if dataset_type == "virat":
		videos_original_dir = os.path.join(data_dir, "VIRAT Ground Dataset", "videos_original")
		video_path = os.path.join(videos_original_dir, video_name+".mp4")
		cap = cv2.VideoCapture(video_path)
		frame_no = 0
		while True:
			ret, frame = cap.read()
			if ret is False:
				break
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
			yield (frame_no, frame)
			frame_no += 1
		cap.release()
	elif dataset_type == "ilsvrc":
		vid_train_dir = os.path.join(data_dir, 'ILSVRC2015', 'Data', 'VID', 'train')
		for	sequences_dir_name in os.listdir(vid_train_dir):
			sequences_dir = os.path.join(vid_train_dir, sequences_dir_name)
			if video_name in os.listdir(sequences_dir):
				frames_path = os.path.join(sequences_dir, video_name)
				break
				
		max_frame_no = -1
		for filename in os.listdir(frames_path):
			if ".JPEG" in filename:
				frame_no = int(filename.split(".JPEG")[0])
				if frame_no > max_frame_no:
					max_frame_no = frame_no
		frame_no = 0
		while frame_no <= max_frame_no:
			frame = cv2.imread(os.path.join(frames_path, "{:06d}.JPEG".format(frame_no)))
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
			yield (frame_no, frame)
			frame_no += 1


def process_vgg_data(video_name, vgg_registered, vgg_data, processed_data):
	vgg_data = np.asarray(vgg_data)
	out = vgg_model.predict(preprocess_input(vgg_data))
	for register_idx, register in enumerate(vgg_registered):
		if register[0] == 'object':
			obj_id = register[1]
			if obj_id in processed_data['object']:
				processed_data['object'][obj_id][0] += out[register_idx]
				processed_data['object'][obj_id][1] += 1
			else:
				processed_data['object'][obj_id] = []
				processed_data['object'][obj_id].append(out[register_idx].copy())
				processed_data['object'][obj_id].append(1)
		else: #object_sampled, object_mask, scene
			with open(register_to_path(video_name, register), 'wb') as f:
				pickle.dump(out[register_idx], f)
	

def process_c3d_data(video_name, c3d_registered, c3d_data, processed_data):
	c3d_data = np.asarray(c3d_data)[..., ::-1]
	out = c3d_model.predict(c3d_data-c3d_mean_cube)
	
	for register_idx, register in enumerate(c3d_registered):
		key = register[1:]
		if len(key) == 1:
			key = key[0]
		if key in processed_data[register[0]]:
			processed_data[register[0]][key][0] += out[register_idx]
			processed_data[register[0]][key][1] += 1
		else:
			processed_data[register[0]][key] = []
			processed_data[register[0]][key].append(out[register_idx].copy())
			processed_data[register[0]][key].append(1)
	

def process_processed_data(video_name, processed_data):
	for folder_name in processed_data:
		for key in processed_data[folder_name]:
			register = (folder_name, ) + (key if type(key)==tuple else (key, ))
			path = register_to_path(video_name, register)
			total_feature, total = processed_data[folder_name][key]
			feature = total_feature / total
			with open(path, "wb") as f:
				pickle.dump(feature, f)


def read_video_fps():
	video_fps = {}
	video_fps_path = os.path.join(features_dir, 'video_fps')
	if os.path.exists(video_fps_path):
		with open(video_fps_path, 'r') as f:
			for line in f:
				video_name, fps = line.split()
				video_fps[video_name] = float(fps)
	return video_fps
	

def write_video_fps(video_fps):
	video_fps_path = os.path.join(features_dir, 'video_fps')
	video_fps = sorted(video_fps.items())
	with open(video_fps_path, 'w') as f:
		f.write('\n'.join([' '.join(map(str, row)) for row in video_fps]))


def extract_video_features(video_name, sequences, dataset_type):
	obj_ids = set([])
	for obj1_id, obj2_id in sequences:
		obj_ids.add(obj1_id)
		obj_ids.add(obj2_id)
		
	bb_gt = get_bb_gt(video_name, dataset_type, obj_ids) # [[obj_id, frame_no, x, y, width, height], ...]
	minimum_frame_number, maximum_frame_number = get_minimum_and_maximum_frame_numbers(bb_gt)
	id_to_start_end = get_id_to_start_end(bb_gt, minimum_frame_number, maximum_frame_number)
	frame_to_bb = get_frame_to_bb(bb_gt, minimum_frame_number, maximum_frame_number)
	pair_start_list, pair_end_list = get_pair_start_end_list(id_to_start_end, sequences, minimum_frame_number, maximum_frame_number)
	obj_start_list, obj_end_list = get_obj_start_end_list(id_to_start_end, minimum_frame_number, maximum_frame_number)
	
	frame_gen = frame_generator(video_name, dataset_type)
	fps, frame_width, frame_height = get_video_feature(video_name, dataset_type)
	sample_period = int(fps*sample_period_sec)
	
	video_fps = read_video_fps()
	video_fps[video_name] = fps
	write_video_fps(video_fps)
	
	vgg_registered = []
	vgg_data = []
	c3d_stacks = {'object_blackened':{}, 'pair_blackened':{}, 'pair':{}}
	c3d_registered = []
	c3d_data = []
	processed_data = {'object':{}, 'object_blackened':{}, 'pair_blackened':{}, 'pair':{}}
	current_pairs = set([])
	current_obj_ids = set([])
	
	
	for frame_no, frame in frame_gen:
		if frame_no < minimum_frame_number:
			continue
		if frame_no > maximum_frame_number:
			break
		
		effective_frame_no = frame_no - minimum_frame_number
		
		frame_obj_ids = set(frame_to_bb[effective_frame_no].keys())
		
		for obj_id in obj_start_list[effective_frame_no]:
			current_obj_ids.add(obj_id)
			c3d_stacks['object_blackened'][obj_id] = []
			
		for pair in pair_start_list[effective_frame_no]:
			current_pairs.add(pair)
			c3d_stacks['pair'][pair] = []
			c3d_stacks['pair_blackened'][pair] = []
			
		for obj_id in obj_end_list[effective_frame_no]:
			current_obj_ids.remove(obj_id)
			del c3d_stacks['object_blackened'][obj_id]
		
		for pair in pair_end_list[effective_frame_no]:
			current_pairs.remove(pair)
			del c3d_stacks['pair'][pair]
			del c3d_stacks['pair_blackened'][pair]
		
		#OBJECT VGG FEATURE
		if frame_no % sample_period == 0:
			for obj_id in frame_obj_ids:
				bb = frame_to_bb[effective_frame_no][obj_id]
				x, y, width, height = bb
				vgg_registered.append(('object', obj_id))
				vgg_data.append(resize_frame(frame[y:y+height, x:x+width,:], vgg_input_size))
				
		#OBJECT BLACKENED C3D FEATURE
		for obj_id in current_obj_ids:
			bb_list = []
			if obj_id in frame_obj_ids:
				bb = frame_to_bb[effective_frame_no][obj_id]
				bb_list.append(scale_bb(bb, frame_width, frame_height, c3d_input_size))
			c3d_stacks['object_blackened'][obj_id].append(blacken_frame(resize_frame(frame, c3d_input_size), bb_list))
		
		
		if frame_no % sample_period == 0 and len(current_pairs) > 0:
			for obj_id in frame_obj_ids:
				bb = frame_to_bb[effective_frame_no][obj_id]
				x, y, width, height = bb
				
				#OBJECT SAMPLED VGG FEATURES
				vgg_registered.append(('object_sampled', frame_no, obj_id))
				vgg_data.append(resize_frame(frame[y:y+height, x:x+width,:], vgg_input_size))
				
				#OBJECT MASK VGG FEATURES
				mask = np.zeros((vgg_input_size, vgg_input_size, 3), dtype=np.float32)
				mask_x, mask_y, mask_width, mask_height = scale_bb(bb, frame_width, frame_height, vgg_input_size)
				mask[mask_y:mask_y+mask_height, mask_x:mask_x+mask_width] = 255.0
				vgg_registered.append(('object_mask', frame_no, obj_id))
				vgg_data.append(mask)
				
			#SCENE VGG FEATURES
			vgg_registered.append(('scene', frame_no))
			vgg_data.append(resize_frame(frame, vgg_input_size))
			
		for pair in current_pairs:
			#PAIR C3D FEATURE
			c3d_stacks['pair'][pair].append(resize_frame(frame, c3d_input_size))
			
			#PAIR BLACKENED C3D FEATURE
			bb_list = []
			for obj_id in pair:
				if obj_id in frame_obj_ids:
					bb = frame_to_bb[effective_frame_no][obj_id]
					bb_list.append(scale_bb(bb, frame_width, frame_height, c3d_input_size))
			c3d_stacks['pair_blackened'][pair].append(blacken_frame(resize_frame(frame, c3d_input_size), bb_list))
			
		for feature_type in c3d_stacks:
			for feature_id in c3d_stacks[feature_type]:
				if len(c3d_stacks[feature_type][feature_id]) == 16:
					c3d_registered.append((feature_type,) + (feature_id if type(feature_id)==tuple else (feature_id,)))					
					c3d_data.append(np.asarray(c3d_stacks[feature_type][feature_id]))
					c3d_stacks[feature_type][feature_id] = c3d_stacks[feature_type][feature_id][8:]
		
		while len(vgg_registered) > vgg_batch_size:
			process_vgg_data(video_name, vgg_registered[:vgg_batch_size], vgg_data[:vgg_batch_size], processed_data)
			vgg_registered = vgg_registered[vgg_batch_size:]
			vgg_data = vgg_data[vgg_batch_size:]
		
		while len(c3d_registered) > c3d_batch_size:
			process_c3d_data(video_name, c3d_registered[:c3d_batch_size], c3d_data[:c3d_batch_size], processed_data)
			c3d_registered = c3d_registered[c3d_batch_size:]
			c3d_data = c3d_data[c3d_batch_size:]
	
	process_vgg_data(video_name, vgg_registered, vgg_data, processed_data)
	process_c3d_data(video_name, c3d_registered, c3d_data, processed_data)
	process_processed_data(video_name, processed_data)


def read_finished():
	finished = []
	if os.path.exists('finished.txt'):
		with open('finished.txt', 'r') as f:
			for line in f:
				finished.append(line.strip())
	return finished


def write_finished(finished):
	with open('finished.txt', 'w') as f:
		f.write('\n'.join(finished))


def extract_features():
	refexp_list = read_refexp(os.path.join(data_dir, 'refexp.csv'))
	video_name_to_sequences = {}
	for video_name, obj1_id, obj2_id, _ in refexp_list:	
		if video_name not in video_name_to_sequences:
			video_name_to_sequences[video_name] = set([])
		video_name_to_sequences[video_name].add(tuple(sorted([obj1_id, obj2_id])))
	
	finished = read_finished()
	
	video_names = sorted(video_name_to_sequences.keys())
	
	for video_idx, video_name in enumerate(video_names):
		print('{:d}/{:d} {:s}'.format(video_idx, len(video_names), video_name))
		if video_name in finished:
			continue
		if 'VIRAT' in video_name:	
			dataset_type = 'virat'
		elif 'ILSVRC' in video_name:
			dataset_type = 'ilsvrc'
		else:
			raise NameError('Video from unknown dataset.')
		
		extract_video_features(video_name, list(video_name_to_sequences[video_name]), dataset_type)
		finished.append(video_name)
		write_finished(finished)
		
	black_frame_batch = np.zeros((1, vgg_input_size, vgg_input_size, 3), dtype=np.float32)
	black_frame_vgg = vgg_model.predict(preprocess_input(black_frame_batch))[0]
	with open(os.path.join(features_dir, 'black_frame'), 'wb') as f:
		pickle.dump(black_frame_vgg, f)
	
		
def get_c3d_model():
	model_dir = os.path.join(feature_extractors_dir, 'c3d-keras', 'models')
	model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
	model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')
	model = model_from_json(open(model_json_filename, 'r').read())
	model.load_weights(model_weight_filename)
	return model
	
def get_vgg_model():
	base_model = VGG16(weights='imagenet', include_top=True)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
	return model
				
if __name__ == "__main__":
	data_dir = 'data'
	feature_extractors_dir = 'feature_extractors'
	features_dir = 'features'
	
	#create feature directories
	for path in ['object', 'object_sampled', 'object_mask', 'scene', 'object_blackened', 'pair_blackened', 'pair']:
		if not os.path.exists(os.path.join(features_dir, path)):
			os.makedirs(os.path.join(features_dir, path))
	
	c3d_model = get_c3d_model()
	c3d_model = Model(inputs=c3d_model.input, outputs=c3d_model.get_layer('fc6').output)
	c3d_mean_cube = np.load(os.path.join(feature_extractors_dir, 'c3d-keras', 'models', 'train01_16_128_171_mean.npy'))
	c3d_mean_cube = np.transpose(c3d_mean_cube, (1, 2, 3, 0))
	c3d_mean_cube = c3d_mean_cube[:, 8:120, 30:142, :] #center crop
	
	vgg_model = get_vgg_model()
	
	vgg_input_size = 224
	c3d_input_size = 112
	
	sample_period_sec = 1 #in seconds
	
	vgg_batch_size = 128
	c3d_batch_size = 128
	
	extract_features()

