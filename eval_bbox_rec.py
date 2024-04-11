#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   eval_bbox_rec.py
@Time    :   2024/01/23 12:34:41
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   modified from https://github.com/shikras/shikra/blob/main/single_image_dataset/rec.py
usage:
for sub in 1 2 5 7
do
    python eval_bbox_rec.py --path_out "../umbrae/evaluation/bbox_results/brainx/sub0${sub}_dim1024"
done
'''

import os
import json
import argparse
from typing import Dict, Any, Sequence
import torch
from torchvision.ops import box_iou #, generalized_box_iou, distance_box_iou, complete_box_iou

def calculate_iou(box1, box2, box_iou_type=box_iou, threshold=0.5):
        ious = box_iou_type(box1 * 1000, box2 * 1000)
        ious = torch.einsum('i i -> i', ious)  # take diag elem
        iou = ious.mean().item()
        correct = (ious > threshold).sum().item()
        accuracy = 1.0 * correct / len(box1)
        return iou, accuracy

def calculate_metric(preds: Sequence[str], targets: Sequence[str], threshold: float = 0.5) -> Dict[str, Any]:
    failed = 0
    target_failed = 0

    pred_boxes, target_boxes = [], []
    for pred, target in zip(preds, targets):
        extract_pred = pred
        extract_target = target
        
        if extract_target is None:
            target_failed += 1
            # print(f"failed to extract ans for target: {target}")
            continue
        if extract_pred is None:
            failed += 1
            # print(f"failed to extract ans for pred: {pred}")
            extract_pred = [0, 0, 0, 0]
        target_boxes.append(extract_target)
        pred_boxes.append(extract_pred)

    with torch.no_grad():
        target_boxes = torch.tensor(target_boxes)
        pred_boxes = torch.tensor(pred_boxes)
        # iou, iou_accuracy = calculate_iou(pred_boxes, target_boxes, box_iou_type=box_iou)
        # 
        # normalized box value is too small, so that the area is 0.
        ious = box_iou(pred_boxes * 1000, target_boxes * 1000)
        ious = torch.einsum('i i -> i', ious)  # take diag elem
        # NOTE: please note iou only calculate for success target
        iou = ious.mean().item()
        correct = (ious > threshold).sum().item()
    return {
        'accuracy': 1.0 * correct / len(targets),
        'iou': iou,
        'target_failed': target_failed,
        'failed': failed
    }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_coco_gt', type=str, default='bbox/coco_bbox_categorized.json')
    parser.add_argument('--path_coco_categorized_label', type=str, default='processing/coco/coco_categorized_labels.json')
    parser.add_argument('--path_out', type=str, default='rec_results/brainx/sub01_dim1024')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    # create global variables without the args prefix
    for attribute_name in vars(args).keys():
        globals()[attribute_name] = getattr(args, attribute_name)
        
    with open(path_coco_categorized_label, 'r') as file:
        coco_cats = json.load(file)

    path_model_out = os.path.join(path_out, 'rec_response.json')

    all_results = {}    
    for inc_class in coco_cats.keys():
        print("calculating for categories: ", inc_class)

        inc_cats = coco_cats[inc_class]

        with open(path_coco_gt, 'r') as file:
            coco_gt = json.load(file)
            # save coco_gt in a list, only one box for each category
            bbox_gt = []
            for image_idx in coco_gt.keys():
                bbox = []
                for category in coco_gt[image_idx].keys() & inc_cats:
                    # print(category)
                    bbox.append(coco_gt[image_idx][category][0])
                bbox_gt.append(bbox)

        # process converted model response  
        with open(path_model_out, 'r') as file:
            model_out = json.load(file)
            # save model_out in a list, only one box for each category
            bbox_pd = []
            for out_id in model_out.keys():
                bbox = []
                for category in model_out[out_id].keys() & inc_cats:
                    if len(model_out[out_id][category]) == 0:
                        bbox_o = None
                    else: 
                        bbox_o = model_out[out_id][category][0][0]
                    bbox.append(bbox_o)
                bbox_pd.append(bbox)

        assert len(bbox_gt) == len(bbox_pd)

        f_bbox_gt = [item for sublist in bbox_gt for item in sublist]
        f_bbox_pd = [item for sublist in bbox_pd for item in sublist]

        result_dict = calculate_metric(f_bbox_pd, f_bbox_gt, threshold)

        # print and save output evaluation scores
        for metric, score in result_dict.items():
            print(f'{metric}: {score}')

        save_path = os.path.join(path_out, f'evaluation_scores_{threshold}.txt')
        with open(save_path, 'a') as file:
            file.write(f'categories ({inc_class}): {inc_cats}\n')
            for metric, score in result_dict.items():
                file.write(f'{metric}: {score}\n')
        print(f'evaluation scores saved to {save_path}')

        # save results
        all_results[inc_class] = result_dict

    markdown_table  = "| acc@0.5 (A) | IoU (A) | acc@0.5 (S) | IoU (S) | acc@0.5 (SC) | IoU (SC) | acc@0.5 (SO) | IoU (SO) | acc@0.5 (IO) | IoU (IO) |\n"
    markdown_table += "|-------------|---------|-------------|---------|--------------|----------|--------------|----------|--------------|----------|\n"
    row = "|"
    for metrics in all_results.values():
        row += f"  {metrics['accuracy']:.4f}     | {metrics['iou']:.4f}  |" 
    markdown_table += row
    markdown_table
    print(markdown_table)