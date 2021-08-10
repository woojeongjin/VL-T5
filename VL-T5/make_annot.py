import json
import h5py
import csv

def make_json(infile_h5, anno, outfile):
    f = h5py.File(infile_h5, 'r')
    ids = set([key for key in f.keys()])
    
    print(len(ids))
    
    with open(anno, 'r') as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            if row[0] in ids:
                annot.append({"img_id": row[0],  "sentf":{"cc": [row[2]]}})

    return annot
    


# make_json('/home/woojeong/VL-T5/datasets/conceptual_captions/features/valid_obj36.h5', '/home/woojeong/VL-T5/datasets/conceptual_captions/annotations/val_imageId2Ann.tsv', 'cc_valid.json')
annot = []
annot = make_json('/mnt/root/vlt5/datasets/conceptual_captions/features/train_obj36.h5', '/mnt/root/vlt5/datasets/conceptual_captions//annotations/train_imageId2Ann.tsv', annot)

with open('/mnt/root/vlt5/datasets/conceptual_captions/annotations/cc_train.json', "w", encoding='utf8') as f:
    json.dump(annot, f, indent=4, sort_keys=True  )
