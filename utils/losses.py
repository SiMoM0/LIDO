# Code from https://github.com/PRBonn/ContMAV/blob/master/src/utils.py

import os
import sys

# import pandas as pd
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, emb_k, emb_q, labels, epoch, tau=0.1):
        """
        emb_k: the feature bank with the aggregated embeddings over the iterations
        emb_q: the embeddings for the current iteration
        labels: the correspondent class labels for each sample in emb_q
        """
        if epoch:
            total_loss = torch.tensor(0.0).cuda()
            assert (
                emb_q.shape[0] == labels.shape[0]
            ), "mismatch on emb_q and labels shapes!"
            emb_k = F.normalize(emb_k, dim=-1)
            emb_q = F.normalize(emb_q, dim=1)

            for i, emb in enumerate(emb_q):
                label = labels[i]
                if not (255 in label.unique() and len(label.unique()) == 1):
                    label[label == 255] = self.n_classes
                    label_sq = torch.unique(label, return_inverse=True)[1]
                    oh_label = (F.one_hot(label_sq)).unsqueeze(-2)  # one hot labels
                    count = oh_label.view(-1, oh_label.shape[-1]).sum(
                        dim=0
                    )  # num of pixels per cl
                    pred = emb.permute(1, 2, 0).unsqueeze(-1)
                    oh_pred = (
                        pred * oh_label
                    )  # (H, W, Nc, Ncp) Ncp num classes present in the label
                    oh_pred_flatten = oh_pred.view(
                        oh_pred.shape[0] * oh_pred.shape[1],
                        oh_pred.shape[2],
                        oh_pred.shape[3],
                    )
                    res_raw = oh_pred_flatten.sum(dim=0) / count  # avg feat per class
                    res_new = (res_raw[~res_raw.isnan()]).view(
                        -1, self.n_classes
                    )  # filter out nans given by intermediate classes (present because of oh)
                    label_list = label.unique()
                    if self.n_classes in label_list:
                        label_list = label_list[:-1]
                        res_new = res_new[:-1, :]

                    # temperature-scaled cosine similarity
                    final = (res_new.cuda() @ emb_k.T.cuda()) / 0.1

                    loss = F.cross_entropy(final, label_list)
                    total_loss += loss

            return total_loss / emb_q.shape[0]

        return torch.tensor(0).cuda()

class VoxelContrastiveLoss(nn.Module):
    def __init__(self, n_classes=19, tau=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.tau = tau

    def forward(self, emb_k, emb_q, labels, epoch):
        """
        emb_k: (C, C) - class prototypes
        emb_q: (N, C) - per-voxel embeddings
        labels: (N,) - voxel-level labels (ignore 0)
        """
        if epoch:
            # move to device
            emb_k = emb_k.cuda()
            # mask out invalid labels
            valid_mask = (labels != 0) & (labels != 255)
            emb_q = emb_q[valid_mask]  # (M, C)
            labels = labels[valid_mask]  # (M,)

            if emb_q.shape[0] == 0:
                return torch.tensor(0.0, device=emb_q.device)

            # Normalize embeddings
            emb_q = F.normalize(emb_q, dim=1)  # (M, C)
            emb_k = F.normalize(emb_k, dim=1)  # (C, C)

            # Compute cosine similarity: (M, C) @ (C, C) -> (M, C)
            logits = emb_q @ emb_k.T  # (M, C)
            logits = logits / self.tau

            # Cross entropy between predicted logits and ground-truth labels
            loss = F.cross_entropy(logits, labels, ignore_index=0)

            return loss

        return torch.tensor(0.0).cuda()

# Proposed confidence-based prototype loss (adapted from ContMAV loss)
class OWLoss(nn.Module):
    def __init__(self, n_classes, hinged=False, delta=0.1, distance='cosine'):
        super().__init__()
        self.n_classes = n_classes
        self.hinged = hinged
        self.delta = delta
        self.distance = distance
        self.count = torch.zeros(self.n_classes).cuda()  # count for class
        self.features = {
            i: torch.zeros(self.n_classes, dtype=torch.float32).cuda() for i in range(self.n_classes)
        }
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # for implementation of Welford Alg.
        self.ex = {i: torch.zeros(self.n_classes, dtype=torch.float32).cuda() for i in range(self.n_classes)}
        self.ex2 = {
            i: torch.zeros(self.n_classes, dtype=torch.float32).cuda() for i in range(self.n_classes)
        }
        self.var = {
            i: torch.zeros(self.n_classes, dtype=torch.float32).cuda() for i in range(self.n_classes)
        }

        if self.distance == 'l1':
            self.criterion = torch.nn.L1Loss(reduction="none")
        elif self.distance == 'l2':
            self.criterion = torch.nn.MSELoss(reduction="none")
        elif self.distance == 'cosine':
            self.criterion = torch.nn.CosineEmbeddingLoss(reduction="none")
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")

        self.previous_features = None
        self.previous_count = None

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        gt_labels = torch.unique(sem_gt).tolist()
        logits_permuted = logits.float() # avoid overflow if using fp16
        for label in gt_labels:
            if label == 255 or label == 0:
                continue
            sem_gt_current = sem_gt == label
            sem_pred_current = sem_pred == label
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            if tps_current.sum() < 5: # TODO this can be changed
                continue
            logits_tps = logits_permuted[torch.where(tps_current == 1)]
            # weight features with point confidence
            conf = logits_permuted[torch.where(tps_current == 1)]
            conf = torch.softmax(conf, dim=1).max(dim=1)[0]
            weights = conf / conf.max()
            logits_tps = logits_tps * weights.unsqueeze(1)
            avg_mav = logits_tps.sum(dim=0, keepdim=True) / weights.sum()
            # max_values = logits_tps[:, label].unsqueeze(1)
            # logits_tps = logits_tps / max_values
            # avg_mav = torch.mean(logits_tps, dim=0)
            n_tps = logits_tps.shape[0]
            # features is running mean for mav
            self.features[label] = (
                self.features[label] * self.count[label] + avg_mav * n_tps
            )

            self.ex[label] += (logits_tps).sum(dim=0)
            self.ex2[label] += ((logits_tps) ** 2).sum(dim=0)
            self.count[label] += n_tps
            self.features[label] /= self.count[label] + 1e-8

    def forward(
        self, logits: torch.Tensor, sem_gt: torch.Tensor, is_train: bool
    ) -> torch.Tensor:
        if is_train:
            # update mav only at training time
            sem_gt = sem_gt.type(torch.uint8)
            self.cumulate(logits, sem_gt)
        if self.previous_features == None:
            return torch.tensor(0.0).cuda()
        gt_labels = torch.unique(sem_gt).tolist()

        logits_permuted = logits

        acc_loss = torch.tensor(0.0).cuda()
        for label in gt_labels[1:]:  # skip 0
            mav = self.previous_features[label]
            logs = logits_permuted[torch.where(sem_gt == label)]
            mav = mav.expand(logs.shape[0], -1)
            if self.previous_count[label] > 0:
                if self.distance == 'cosine':
                    ew_l1 = self.criterion(logs, mav, torch.ones(logs.shape[0], device=logs.device))
                else:
                    ew_l1 = self.criterion(logs, mav)
                # original
                # ew_l1 = ew_l1 / (self.var[label] + 1e-8) # maybe add epsilon to avoid division by very small numbers
                # modified
                # safe_var = torch.clamp(self.var[label], min=1e-2)
                # ew_l1 = ew_l1 / safe_var
                if self.hinged:
                    ew_l1 = F.relu(ew_l1 - self.delta).sum(dim=1)
                acc_loss += ew_l1.mean()

        return acc_loss

    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count
        for c in self.var.keys():
            self.var[c] = (self.ex2[c] - self.ex[c] ** 2 / (self.count[c] + 1e-8)) / (
                self.count[c] + 1e-8
            )

        # resetting for next epoch
        self.count = torch.zeros(self.n_classes)  # count for class
        self.features = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }
        self.ex = {i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)}
        self.ex2 = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }

        return self.previous_features, self.var

    def read(self):
        mav_tensor = torch.zeros(self.n_classes, self.n_classes)
        for key in self.previous_features.keys():
            mav_tensor[key] = self.previous_features[key]
        return mav_tensor


class ObjectosphereLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, logits, sem_gt):
        logits_unk = logits[torch.where(sem_gt == 255)]
        logits_kn = logits[torch.where(sem_gt != 255)]

        if len(logits_unk):
            loss_unk = (torch.linalg.norm(logits_unk, dim=1)**2).mean()
        else:
            loss_unk = torch.tensor(0)
        if len(logits_kn):
            loss_kn = F.relu(self.sigma - (torch.linalg.norm(logits_kn, dim=1)**2)).mean()
        else:
            loss_kn = torch.tensor(0)

        loss = 10 * loss_unk + loss_kn # NOTE: this parameter can be tuned (10 as default)
        return loss


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, device, weight):
        super(CrossEntropyLoss2d, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction="none",
            ignore_index=-1,
        )
        self.ce_loss.to(device)

    def forward(self, inputs, targets):
        losses = []
        targets_m = targets.clone()
        # if targets_m.sum() == 0:
        #     import ipdb;ipdb.set_trace()  # fmt: skip
        targets_m -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        number_of_pixels_per_class = torch.bincount(
            targets.flatten().type(self.dtype), minlength=self.num_classes
        )
        divisor_weighted_pixel_sum = torch.sum(
            number_of_pixels_per_class[1:] * self.weight
        )  # without void
        if divisor_weighted_pixel_sum > 0:
            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
        else:
            losses.append(torch.tensor(0.0).cuda())

        return losses


class CrossEntropyLoss2dForValidData:
    def __init__(self, device, weight, weighted_pixel_sum):
        super(CrossEntropyLoss2dForValidData, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(), reduction="sum", ignore_index=-1
        )
        self.ce_loss.to(device)
        self.weighted_pixel_sum = weighted_pixel_sum
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss

    def compute_whole_loss(self):
        return self.total_loss.cpu().numpy().item() / self.weighted_pixel_sum.item()

    def reset_loss(self):
        self.total_loss = 0


class CrossEntropyLoss2dForValidDataUnweighted:
    def __init__(self, device):
        super(CrossEntropyLoss2dForValidDataUnweighted, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=None, reduction="sum", ignore_index=-1
        )
        self.ce_loss.to(device)
        self.nr_pixels = 0
        self.total_loss = 0

    def add_loss_of_batch(self, inputs, targets):
        targets_m = targets.clone()
        targets_m -= 1
        loss = self.ce_loss(inputs, targets_m.long())
        self.total_loss += loss
        self.nr_pixels += torch.sum(targets_m >= 0)  # only non void pixels

    def compute_whole_loss(self):
        return (
            self.total_loss.cpu().numpy().item() / self.nr_pixels.cpu().numpy().item()
        )

    def reset_loss(self):
        self.total_loss = 0
        self.nr_pixels = 0