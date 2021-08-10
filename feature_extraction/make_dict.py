import sys
import csv
import base64
import time
from tqdm import tqdm
import numpy as np
import h5py
import argparse
import json

def make_json(args, anno):
	ids = set()
	path = args.h5_path
	pointer = dict()

	print('-----reading h5py')
	for d in tqdm(range(24)):
		path_name = path+str(d)+'.h5'
		f = h5py.File(path+str(d)+'.h5', 'r')

		file_name = path_name.split('/')[-1]
		ids.update(set([key for key in f.keys()]))

		keys = [key for key in f.keys()]
		for key in keys:
			pointer[key] = file_name

	print(len(ids))
	with open('/mnt/root/vlt5/datasets/lxmert/cc_train_pointer_h5.json', "w", encoding='utf8') as f:
	    json.dump(pointer, f)


	
	annot = []
	print('writing ids')
	with open(anno, 'r') as fd:
	    rd = csv.reader(fd, delimiter="\t", quotechar='"')
	    for row in tqdm(rd):
	        if row[0] in ids:
	            annot.append({"img_id": row[0],  "sentf":{"cc": [row[2]]}})

	with open('/mnt/root/vlt5/datasets/lxmert/cc_train.json', "w", encoding='utf8') as f:
	    json.dump(annot, f, indent=4, sort_keys=True )


	print('done make json and ointer')


def make_pointer(args):

	pointer = dict()
	path = args.h5_path
	print('making pointer...')
	for d in tqdm(range(24)):
		path_name = path+str(d)+'.h5'
		f = h5py.File(path_name, 'r')

		file_name = path_name.split('/')[-1]
		keys = [key for key in f.keys()]
		for key in keys:
			pointer[key] = file_name

	




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--h5_path', type=str,
                        default='/mnt/root/vlt5/datasets/conceptual_captions/features/train_obj36_split_attr_')

	args = parser.parse_args()



	make_json(args,'/mnt/root/vlt5/datasets/conceptual_captions/annotations/train_imageId2Ann.tsv')
	# make_pointer(args)