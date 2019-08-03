import os
import csv
import random

train_split = 0.7
val_split = 0.857


def sort_wrt_video_idx(refexp_list, video_list):
	video_name2idx = {}
	for video_idx, video_name in enumerate(video_list):
		video_name2idx[video_name] = video_idx
	
	new_refexp_list = []
	for video_name, obj1, obj2, refexp in refexp_list:
		new_refexp_list.append((video_name2idx[video_name], (video_name, obj1, obj2, refexp)))

	refexp_list = sorted(new_refexp_list)

	new_refexp_list = []
	for video_idx, sequence in refexp_list:
		new_refexp_list.append(sequence)
	
	return new_refexp_list
	

refexp_list = []
video_set = set([])

with open(os.path.join('data', 'refexp.csv'), 'r', newline='') as csv_file:
	reader = csv.reader(csv_file)
	for row in reader:
		video_name = row[0]
		obj1_id = int(row[1])
		obj2_id = int(row[2])
		refexp = row[3]
		if len(refexp.split()) < 25:
			refexp_list.append([video_name, obj1_id, obj2_id, refexp])
			video_set.add(video_name)
	
video_list = list(video_set)
random.shuffle(video_list)

refexp_list = sort_wrt_video_idx(refexp_list, video_list)

split_index = int(len(refexp_list)*train_split)
split_video_name = refexp_list[split_index][0]

while refexp_list[split_index][0] == split_video_name:
	split_index += 1

train_sequences = refexp_list[:split_index]
test_sequences = refexp_list[split_index:]

random.shuffle(train_sequences)

train_val_sequences = train_sequences
split_index = int(len(train_val_sequences)*val_split)

train_sequences = train_val_sequences[:split_index]
val_sequences = train_val_sequences[split_index:]

train_sequences = sort_wrt_video_idx(train_sequences, video_list)
val_sequences = sort_wrt_video_idx(val_sequences, video_list)
		
with open(os.path.join('dataset_split', 'train_refexp.csv'), 'w', newline='') as csv_file:
	writer = csv.writer(csv_file)
	for row in train_sequences:
		writer.writerow(row)

with open(os.path.join('dataset_split', 'val_refexp.csv'), 'w', newline='') as csv_file:
	writer = csv.writer(csv_file)
	for row in val_sequences:
		writer.writerow(row)
		
with open(os.path.join('dataset_split', 'test_refexp.csv'), 'w', newline='') as csv_file:
	writer = csv.writer(csv_file)
	for row in test_sequences:
		writer.writerow(row)

print('Dataset is split into train, validation and test. Saved to {:s}.'.format(os.path.join('viref', 'dataset_split')))

