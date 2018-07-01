from __future__ import division

from .models import Darknet
from .utils.parse_config import parse_data_config
from .utils.utils import non_max_suppression
from .utils.utils import bbox_iou
from .utils.utils import compute_ap
from .utils.datasets import ListDataset

import torch
from torch.autograd import Variable
import numpy as np


def test(model_config_path,
         data_config_path,
         weights_path,
         batch_size=16,
         iou_thres=0.5,
         conf_thres=0.5,
         nms_thres=0.45,
         n_cpu=0,
         img_size=416,
         use_cuda=True,
         verbose=False):
    cuda = torch.cuda.is_available() and use_cuda

    # Get data configuration
    data_config = parse_data_config(data_config_path)
    test_path = data_config['valid']

    # Initiate model
    model = Darknet(model_config_path)
    model.load_weights(weights_path)

    if cuda:
        model = model.cuda()

    model.eval()

    # Get dataloader
    dataset = ListDataset(test_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    correct = 0

    print('Compute mAP...')

    targets = None
    APs = []
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = targets.type(Tensor)

        with torch.no_grad():
            output = model(imgs)
            # TODO: parametrize the number of classes
            output = non_max_suppression(
                output, 80, conf_thres=conf_thres, nms_thres=nms_thres)

        # Compute average precision for each sample
        for sample_i in range(targets.size(0)):
            correct = []

            # Get labels for sample where width is not zero (dummies)
            annotations = targets[sample_i, targets[sample_i, :, 3] != 0]
            # Extract detections
            detections = output[sample_i]

            if detections is None:
                # If there are no detections but there are annotations mask as zero AP
                if annotations.size(0) != 0:
                    APs.append(0)
                continue

            # Get detections sorted by decreasing confidence scores
            detections = detections[np.argsort(-detections[:, 4])]

            # If no annotations add number of detections as incorrect
            if annotations.size(0) == 0:
                correct.extend([0 for _ in range(len(detections))])
            else:
                # Extract target boxes as (x1, y1, x2, y2)
                target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
                target_boxes[:, 0] = (
                    annotations[:, 1] - annotations[:, 3] / 2)
                target_boxes[:, 1] = (
                    annotations[:, 2] - annotations[:, 4] / 2)
                target_boxes[:, 2] = (
                    annotations[:, 1] + annotations[:, 3] / 2)
                target_boxes[:, 3] = (
                    annotations[:, 2] + annotations[:, 4] / 2)
                target_boxes *= img_size

                detected = []
                for *pred_bbox, conf, obj_conf, obj_pred in detections:
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # Compute iou with target boxes
                    iou = bbox_iou(pred_bbox, target_boxes)
                    # Extract index of largest overlap
                    best_i = np.argmax(iou)
                    # If overlap exceeds threshold and classification is correct mark as correct
                    if iou[best_i] > iou_thres and obj_pred == annotations[
                            best_i, 0] and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # Extract true and false positives
            true_positives = np.array(correct)
            false_positives = 1 - true_positives

            # Compute cumulative false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # Compute recall and precision at all ranks
            recall = true_positives / annotations.size(0) if annotations.size(
                0) else true_positives
            precision = true_positives / np.maximum(
                true_positives + false_positives, np.finfo(np.float64).eps)

            # Compute average precision
            AP = compute_ap(recall, precision)
            APs.append(AP)

            if verbose is True:
                print("+ Sample [%d/%d] AP: %.4f (%.4f)" %
                    (len(APs), len(dataset), AP, np.mean(APs)))

    print("Mean Average Precision: %.4f" % np.mean(APs))
    return np.mean(APs)
