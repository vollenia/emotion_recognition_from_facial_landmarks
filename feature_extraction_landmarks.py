# Importing modules
import os
import argparse
import time
import pandas as pd
import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import FileVideoStream

# Argparse constructor
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", required=True,
	help = "path to the data file directory")
parser.add_argument("-l", "--labels", required=True,
	help = "path to the labels")
parser.add_argument("-f", "--frames", required=True, type=int,
	help = "Nr of frames to be extractd")
parser.add_argument("-p", "--predictor", required=True,
	help = "path to the landmark predictor")
args = vars(parser.parse_args())

# Extracting features
# Target frames
def frame_selection(frames, nr):
	vip_frames = np.array([frame for frame in frames[:len(frames)//nr*nr:len(frames)//nr]])
	
	return vip_frames # returns a 3D array

def process_video(files):
	start_time = time.time()
	data = {}
	for file_path in files:
		cap = FileVideoStream(file_path).start()
		frames = []
		while cap.more():
			try:
				frame = cap.read()
				# Reduces the res of the file for faster processing 
				frame = imutils.resize(frame, width=400)
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				rects = detector(frame, 0)
				if len(rects) > 0:  # if at least one face detected
					face = rects[0]
					shape = predictor(frame, face)
					shape = face_utils.shape_to_np(shape)
					# Checking if landmark coordinates contain negative values
					neg = np.where(shape < 0)
					if len(neg[0]) == 0:
						frames.append(shape)
					else:
						print(f'negative coordinates found for {file_path}')
			except AttributeError:
				pass
		cap.stop()
		if len(frames) >= args["frames"]: #checks if enough frames were accumulated
			data[file_path.split('/')[-1].replace('.avi', '')] = frame_selection(frames, args["frames"])# nr of target framems
		else:
			data["Y"+file_path.split('/')[-1].replace('.avi', '')] = None
	end_time = time.time()
	duration = end_time - start_time
	print(f'video processing took {duration} seconds')
	return data

# Extracting labels
def get_labels(file):
	start_time = time.time()
	ref = ["anger", "happiness", "neutral", "sadness"]
	labels = {}
	nr_l = 0
	nr_bl = 0
	with open(file, 'r') as f:
		for line in f:
			inner_d = {}
			splt = line.split(';')
			if splt[1] in ref: # filtering down to 4 calsses
				inner_d["LABEL"] = splt[1]
				inner_d["ACTIVATION"] = float(splt[2][2:])
				inner_d["VALENCE"] = float(splt[3][2:])
				labels[splt[0]] = inner_d
			else:
				labels["X"+splt[0]] = None
				nr_bl +=1
			nr_l += 1
	end_time = time.time()
	duration = end_time - start_time
	print(f'label processing of {nr_l} line(s) ({nr_bl} bad labels detected) took {duration} seconds')
	return labels

# Creating dataset as dictionary
def create_dataset(labels, data):
	start_time = time.time()
	dataset_d = {}
	nr_l = 0
	nr_bl = 0
	for x, y, i, j  in zip(labels.keys(), data.keys(), labels.values(), data.values()):
		if x == y:
			features = {}
			features["FEATURES"]= j
			i.update(features)
			dataset_d[x] = i
		else:
			print(f'filtered out: {x} {y}')
			nr_bl += 1
		nr_l += 1
	end_time = time.time()
	duration = end_time - start_time
	print(f'combining of {nr_l} line(s) ({nr_bl} lines filtered) took {duration} seconds')
	return dataset_d

if __name__ == '__main__':
	# Creating a list with paths from directory
	list_of_files = sorted([os.path.join(args["data"], i) for i in os.listdir(args["data"])])

	# Defining face detector & landmarks pedictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["predictor"])

	# Extracting features
	data = process_video(list_of_files)

	# Extracting labels
	labels = get_labels(args["labels"])

	dataset_d = create_dataset(labels, data)
	print(f'original length of dataset was: {len(data)} VS new length of dataset is: {len(dataset_d)}')

	# Creating pandas dataframe from dictionary
	dataset_df = pd.DataFrame.from_dict(dataset_d, orient='index')

	# Storing the dataframe
	dataset_df.to_pickle("df_hog_landmarks"+str(args["frames"])+".pkl")