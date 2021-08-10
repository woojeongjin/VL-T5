# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
from tqdm import tqdm
import numpy as np
import h5py
import argparse
import json

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname,  data, topk=None):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    # data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in tqdm(enumerate(reader), ncols=150):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            skip = False
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(
                    base64.b64decode(item[key]), dtype=dtype)
                if len(item[key]) % shape[0]!= 0:
                    skip=True
                    print('cannot reshape')
                else:
                    item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)
            if not skip:
                data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." %
          (len(data), fname, elapsed_time))
    return data

def make_json(infile_h5, anno, annot):
    f = h5py.File(infile_h5, 'r')
    ids = set([key for key in f.keys()])
    
    print(len(ids))
    
    with open(anno, 'r') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if row[0] in ids:
                annot.append({"img_id": row[0],  "sentf":{"cc": [row[2]]}})

    return annot

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_path', type=str,
                        default='/mnt/root/vlt5/datasets/conceptual_captions/train_feat_attr/train_obj36_')
    parser.add_argument('--h5_path', type=str,
                        default='/mnt/root/vlt5/datasets/conceptual_captions/features/train_obj36_split_attr_')

    parser.add_argument('--id', type=int,
                        default=0)

    args = parser.parse_args()
    dim = 2048

    print('Load ', args.tsv_path)
    # data = []
    # for i in range(48):
    #     data = load_obj_tsv(args.tsv_path+str(i)+'.tsv', data)
    # print('# data:', len(data))

    output_fname = args.h5_path+ str(args.id) + '.h5'


    print('features will be saved at', output_fname)


    ids = list(range(48))
    ids = [ids[i::24] for i in range(24)][args.id]
    for ii in ids:
        data = load_obj_tsv(args.tsv_path+str(ii)+'.tsv', [])
        with h5py.File(output_fname, 'a') as f:
            for i, datum in tqdm(enumerate(data),
                                ncols=150,):

                img_id = datum['img_id']

                num_boxes = datum['num_boxes']
                if num_boxes != 36:
                    continue

                grp = f.create_group(img_id)
                grp['features'] = datum['features'].reshape(num_boxes, 2048)
                grp['obj_id'] = datum['objects_id']
                grp['obj_conf'] = datum['objects_conf']
                grp['attr_id'] = datum['attrs_id']
                grp['attr_conf'] = datum['attrs_conf']
                grp['boxes'] = datum['boxes']
                grp['img_w'] = datum['img_w']
                grp['img_h'] = datum['img_h']


        


    # make_json('/home/woojeong/VL-T5/datasets/conceptual_captions/features/valid_obj36.h5', '/home/woojeong/VL-T5/datasets/conceptual_captions/annotations/val_imageId2Ann.tsv', 'cc_valid.json')
    # annot = []

    # annot = make_json('/mnt/root/vlt5/datasets/conceptual_captions/features/train_obj36_attr.h5', '/mnt/root/vlt5/datasets/conceptual_captions/annotations/train_imageId2Ann.tsv', annot)

    # with open('/mnt/root/vlt5/datasets/conceptual_captions/annotations/cc_train_attr.json', "w", encoding='utf8') as f:
    #     json.dump(annot, f, indent=4, sort_keys=True )