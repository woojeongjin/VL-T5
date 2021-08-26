import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.models as models

from modeling_gpt2 import FROZEN


def get_visn_arch(arch):
    try:
        return getattr(models, arch)
    except AttributeError as e:
        print(e)
        print("There is no arch %s in torchvision." % arch)


class FROZENPretraining(FROZEN):
    def __init__(self, config):
        super().__init__(config)

        self.losses = self.config.losses.split(',')

        self.resnet = get_visn_arch(config.arch)(pretrained=config.pretrained)

        # backbone_dim = self.resnet.fc.in_features

        # self.vae.get_codebook_indices(images) images: (batch, channels (3), h, w)
        # output = (batch, indices)

    def train_step(self, batch):

        device = next(self.parameters()).device
        tensor_img = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        vis_feats = self.resnet(tensor_img)



        lm_labels = batch["target_ids"].to(device)


        loss_weights = batch["loss_weights"].to(device)

        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vis_inputs=vis_feats,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output
        new_labels = lm_labels.clone()[:,1:]
        lm_mask = new_labels != -100
        lm_mask = lm_mask.float()
        B, L = new_labels.size()
    

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        tensor_img = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        

        vis_feats = self.resnet(tensor_img)

        lm_labels = batch["target_ids"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vis_inputs=vis_feats,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        #########
        new_labels = lm_labels.clone()[:,1:]
        lm_mask = new_labels != -100
        lm_mask = lm_mask.float()
        B, L = new_labels.size()
    

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                # result[f'{task}_loss'] = task_loss[task] / task_counts[task]
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
            # else:
            #     result[f'{task}_loss'] = torch.zeros_like(loss)

        if 'qa' in self.losses:
            output = self.generate(
                input_ids=input_ids,
                vis_inputs=vis_feats,
            )

            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

            results['qa_pred'] = generated_sents

        return results

    @torch.no_grad()
    def generate_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        tensor_img = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        vis_feats = self.resnet(tensor_img)

        # vis_attention_mask = None
        # if 'vis_attention_mask' in batch:
        #     vis_attention_mask = batch['vis_attention_mask'].to(device)

        output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vis_inputs=vis_feats,
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return generated_sents

