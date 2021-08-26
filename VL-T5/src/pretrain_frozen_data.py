from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy
import spacy
import re
from spacy.tokenizer import Tokenizer


import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader

from transformers import GPT2TokenizerFast


import preprocess
from qa_answer_table import AnswerTable

# project_dir = Path(__file__).resolve().parent.parent # VLT5
# workspace_dir = project_dir.parent
workspace_dir = Path('/mnt/root/vlt5')
dataset_dir = workspace_dir.joinpath('datasets/').resolve()
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
cc_dir = dataset_dir.joinpath('conceptual_captions')
coco_img_dir = coco_dir.joinpath('images/')


# Load VG Classes
vg_classes = []
with open(vg_dir.joinpath('objects_vocab.txt')) as f:
    for obj in f.readlines():
        vg_classes.append(obj.split(',')[0].lower().strip())

vg_attrs = []
with open(vg_dir.joinpath('attributes_vocab.txt')) as f:
    for attr in f.readlines():
        vg_attrs.append(attr.split(',')[0].lower().strip())

def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx)


def get_datum(datum):
    data = []
    _sents = []

    args = datum['args']

    if datum['is_train']:
        if 'COCO_train2014' in datum['img_id']:
            img_source = 'mscoco_resplit_train_train2014'
        elif 'COCO_val2014' in datum['img_id']:
            img_source = 'mscoco_resplit_train_val2014'
        elif 'cc' in datum['sentf'].keys():
            img_source = 'cc_train'
        else:
            img_source = 'vgnococo'
    else:
        if 'COCO_val2014' in datum['img_id']:
            img_source = 'mscoco_resplit_val'
        elif 'cc' in datum['sentf'].keys():
            img_source = 'cc_valid'
        else:
            print("no!")

    for text_source, sents in datum['sentf'].items():
        if datum['caption_only']:
            if text_source not in ['mscoco', 'vg', 'cc']:
                continue

        if args.coco_only:
            if text_source != 'mscoco':
                continue

        labels = None

        img_id = datum['img_id']

        for sent_idx, sent in enumerate(sents):

            if len(sent.split()) <= 2:
                continue

            # remove duplicate sentence
            if sent in _sents:
                continue

            new_datum = {
                'uid': make_uid(img_id, text_source, sent_idx),
                'img_id': img_id,
                'img_source': img_source,
                'sent': sent,
                'text_source': text_source
            }


            # Task: Language modeling
            if datum['lm'] and labels is None:
                new_datum = deepcopy(new_datum)
                new_datum['task'] = 'lm'
                new_datum['label'] = None
                data.append(new_datum)


            _sents.append(sent)


    # for d in data:
    #     assert 'task' in d

    return data



class PretrainDataset(Dataset):
    def __init__(self, split='vg', rank=-1, topk=-1, verbose=True, args=None, is_train=True):

        self.topk = topk
        self.verbose = verbose
        self.args = args

        
        # self.vae = OpenAIDiscreteVAE()

        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.pipeline = [("tagger", self.nlp.tagger), ("parser", self.nlp.parser)]
        self.nlp.tokenizer = Tokenizer(self.nlp.vocab, token_match=re.compile(r'\S+').match)

        self.pointer_h5 = None



        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if is_train:
            self.img_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ])

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        # Answer Table from LXMERT (Could be removed)
        self.answer_table = AnswerTable()
        if self.verbose:
            print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        self.img_ids_to_source = {}

        losses = args.losses.split(',')

        data = []
        for img_source in self.sources:
            if img_source == 'cc_train':
                with open(dataset_dir.joinpath('lxmert/cc_train_pointer_h5.json')) as f:
                    self.pointer_h5 = json.load(f)


            data_info_path = dataset_dir.joinpath(f'lxmert/{img_source}.json')
            with open(data_info_path) as f:
                _data = json.load(f)
                if self.verbose:
                    print(f"Loaded {len(_data)} data from", img_source)
                # source_img_ids.append([d['img_id'] for d in _data])
                for datum in _data:
                    self.img_ids_to_source[datum['img_id']] = img_source
                    # datum['img_source'] = img_source
                    datum['args'] = args
                    datum['is_train'] = is_train
                    datum['caption_only'] = args.caption_only

                    datum['lm'] = 'lm' in losses
                    # datum['lmnoun'] = 'lmnoun' in losses
                    # datum['qa'] = 'qa' in losses
                    # datum['prefix'] = 'prefix' in losses
                    # datum['ar'] = 'ar' in losses
                    # datum['captioning'] = 'captioning' in losses


                    # datum['ground_caption'] = 'ground_caption' in losses
                    # datum['refer'] = 'refer' in losses
                    # datum['itm'] = 'itm' in losses
                    # datum['caption'] = 'caption' in losses

                    datum['backbone'] = self.args.backbone

                data.extend(_data)


        if self.verbose:
            print("# images:", len(data))

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        if 'qa' in args.losses:
            self.evaluator = QAEvaluator(data)

        with Pool(8) as pool:
            if self.verbose:
                data = [datum for _data in tqdm(
                    pool.imap(get_datum, data), total=len(data), ncols=100, desc="Creating pretrainig data examples") for datum in _data]
            else:
                data = [datum for _data in pool.imap(
                    get_datum, data) for datum in _data]

        if self.args.itm_cocoonly:
            caption_sources = ['mscoco']
        else:
            caption_sources = ['mscoco', 'vg', 'cc']
        self.data_captions = [datum for datum in data if datum['text_source'] in caption_sources]
        self.n_data_captions = len(self.data_captions)

        if self.verbose:
            print('# itm data:', self.n_data_captions)

        self.data = data
        self.n_data = len(self.data)

        if self.verbose and is_train:
            from collections import Counter
            task_counter = Counter()
            for datum in data:
                try:
                    task_counter.update([datum['task']])
                except KeyError:
                    print(datum)
                    exit()

            print(task_counter)
            for k, v in task_counter.items():
                print(k, f'{v/len(data)*100:.1f}%')

        if self.verbose:
            print("# examples:", len(data))


        self.source_to_img = {
            'mscoco_resplit_train_train2014': coco_dir.joinpath('Image').joinpath('train2014'),
            'mscoco_resplit_train_val2014': coco_dir.joinpath('Image').joinpath('val2014'),
            'mscoco_resplit_val': coco_dir.joinpath('Image').joinpath('val2014'),
            'vgnococo': vg_dir.joinpath('VG_100K'),
            'cc_train': cc_dir.joinpath('images').joinpath('train'),
            'cc_valid': cc_dir.joinpath('images').joinpath('val_complete'),

        }

        self.n_boxes = args.n_boxes

        if 'gpt2' in self.args.backbone:
            # self.tokenizer = GPT2TokenizerFast.from_pretrained(
            #     args.backbone, do_lower_case=args.do_lower_case, pad_token='<PAD>')
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                args.backbone, do_lower_case=args.do_lower_case)
            # self.tokenizer.padding_side = 'left'


    def __len__(self):
        # return len(self.data)
        return self.n_data

    def __getitem__(self, idx):

        out_dict = {}
        out_dict['args'] = self.args

        datum = self.data[idx]
        uid = datum['uid']
        out_dict['uid'] = uid

        ###### Image ######
        img_id = datum['img_id']
        source = datum['img_source']

        img_path = self.source_to_img[source].joinpath(img_id+'.jpg')
        try:
            pil_img = default_loader(img_path)
        except Exception as e:
            print(e)
            print(img_path)
            return self.__getitem__((idx + 95) % self.__len__())
        tensor_img = self.img_transform(pil_img)




        # if source == 'cc_train':
        #     path = cc_dir.joinpath('features').joinpath(self.pointer_h5[img_id])
        #     f = h5py.File(path, 'r')
        # else: 
        #     f = self.source_to_h5[source]
        #     if isinstance(f, Path):
        #         path = self.source_to_h5[source]
        #         f = h5py.File(path, 'r')
            # self.source_to_h5[source] = f

        if 'gpt2' in self.args.backbone:

            text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                assert text_source in ["mscoco", 'vg', 'cc']

                # prefix = "span prediction:"
                prefix = ""
                sent = datum['sent']
                # source_text, target_text = preprocess.corrupt_spans(
                #     sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)

                # input_tokens = [source_text]
                # source_text = ' '.join(input_tokens)
            
                source_text = prefix + sent

            eos_token_id = self.tokenizer.eos_token_id

            # inputs_ids = self.tokenizer.encode(source_text, truncation=True, max_length = self.args.max_text_length-1)


            # input_ids = inputs['input_ids'] +[eos_token_id]
            # attention_mask =inputs['attention_mask'] + [1] 
            # start_idx = self.args.max_text_length - sum(attention_mask)
            # attention_mask = attention_mask + [1] * self.args.vis_size

            input_ids = self.tokenizer.encode(
                source_text, truncation=True, max_length=self.args.max_text_length-1) +[eos_token_id]
                

            padded_ids = input_ids  + [eos_token_id] * (self.args.max_text_length - len(input_ids))
            attention_mask = [0] * len(padded_ids)
            attention_mask[:len(input_ids)] = [1] * len(input_ids)
            attention_mask = [1] * self.args.vis_size + attention_mask

            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(padded_ids)
            out_dict['input_length'] = len(padded_ids)
            out_dict['attention_mask'] = torch.LongTensor(attention_mask)
            out_dict['target_ids'] = torch.LongTensor(padded_ids.copy())
            out_dict['target_length'] = len(padded_ids)

            out_dict['source_text'] = source_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight

            # feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            # try:
            #     f[f'{img_id}/features'].read_direct(feats)
            # except KeyError:
            #     print(uid)
            #     print(source)
            #     print(img_id)
            #     exit()

            feats = tensor_img
            out_dict['vis_feats'] = feats

            # Normalize the boxes (to 0 ~ 1)
            # img_h = f[f'{img_id}/img_h'][()]
            # img_w = f[f'{img_id}/img_w'][()]
            # boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            # boxes[:, (0, 2)] /= img_w
            # boxes[:, (1, 3)] /= img_h
            # np.testing.assert_array_less(boxes, 1+1e-5)
            # # np.testing.assert_array_less(boxes, 1+5e-2)
            # np.testing.assert_array_less(-boxes, 0+1e-5)
            # boxes = torch.from_numpy(boxes)
            # boxes.clamp_(min=0.0, max=1.0)
            # out_dict['boxes'] = boxes

            return out_dict



    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        V_L = len(batch[0]['vis_feats'])

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)
        A_W_L = max(len(entry['attention_mask']) for entry in batch)

        feat_dim = batch[0]['vis_feats'].shape[-1]

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.eos_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.eos_token_id
        attention_mask = torch.ones(B, A_W_L, dtype=torch.long) * 0

        # boxes = torch.zeros(B, V_L, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, V_L, feat_dim, feat_dim, dtype=torch.float)

        loss_weights = torch.ones(B, dtype=torch.float)

        sentences = []
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            attention_mask[i, :len(entry['attention_mask'])] = entry['attention_mask']

            # boxes[i] += entry['boxes']
            vis_feats[i] = entry['vis_feats']

            if 'ans' in entry:
                ans.append(entry['ans'])

            if 'task' in entry:
                tasks.append(entry['task'])

            sentences.append(entry['sent'])
            uids.append(entry['uid'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            # if 'target_text' in entry:
            #     target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        # vis_feats = self.vae.get_codebook_indices(vis_feats)

        assert 'gpt2' in args.backbone
        # word_mask = target_ids != self.tokenizer.pad_token_id
        word_mask = attention_mask[:,2:] != 0
        target_ids[~word_mask] = -100

        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        # batch_entry['target_text'] = target_text

        # Todo
        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids
        batch_entry['attention_mask'] = attention_mask

        # Todo
        # batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats

        batch_entry['loss_weights'] = loss_weights

        batch_entry['uid'] = uids
        batch_entry['sent'] = sentences

        return batch_entry


def get_loader(args, split='vgnococo', mode='train',
               batch_size=32, workers=4, distributed=False, is_master=False,
               topk=-1):


    verbose = is_master
    dataset = PretrainDataset(
        split,
        rank=-1,
        topk=topk,
        verbose=verbose,
        args=args,
        is_train=(mode == 'train'),
        )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    return loader


class QAEvaluator:
    def __init__(self, data):

        # Create QA Eval Data
        self.data = []
        for datum in data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        return dset2score, dset2cnt, score, cnt

    def _evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplementedError
