#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   eval_caption.py
@Time    :   2025/03/22 20:23:10
@Author  :   Weihao Xia 
@Version :   2.0
@Desc    :   
usage:
for sub in 1 2 5 7
do
    python eval_caption.py ../umbrae/evaluation/caption_results/brainx/sub0${sub}_dim1024/fmricap.json \
        data/caption/images --references_json data/caption/fmri_cococap.json
done
'''

import os
import json
import clip
import warnings
import argparse
import pathlib
import numpy as np
from PIL import Image
import datetime

import torch
from sentence_transformers import SentenceTransformer
from metrics import extract_all_images, get_all_metrics
from metrics.clip_score import CLIPScore, RefCLIPScore
from metrics.pac_score import PACScore, RefPACScore
from metrics.sentence_score import SentenceScore

_MODELS = {
    "ViT-B/32": "checkpoints/clip_ViT-B-32.pth",
    "open_clip_ViT-L/14": "checkpoints/openClip_ViT-L-14.pth"
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('candidates_json', type=str,default='example/good_captions.json')
    parser.add_argument('image_dir', type=str, default='example/images')
    parser.add_argument('--references_json', default='example/refs.json')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', choices=['ViT-B/32', 'open_clip_ViT-L/14'])
    parser.add_argument('--save_per_instance', default=None, help='save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args

def main():
    args = parse_args()

    image_paths = [os.path.join(args.image_dir, path) for path in os.listdir(args.image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
    image_ids = [pathlib.Path(path).stem for path in image_paths]

    with open(args.candidates_json) as f:
        candidates = json.load(f)
    candidates = [candidates[cid] for cid in image_ids]

    if args.references_json:
        with open(args.references_json) as f:
            references = json.load(f)
            references = [references[cid] for cid in image_ids]
            if isinstance(references[0], str):
                references = [[r] for r in references]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')

    # calculate sentence similarities
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    _, sen_scores = SentenceScore(sentence_model, references, candidates, device)

    # calculate clipscore
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    # get image-text clipscore
    _, clip_scores, candidate_feats = CLIPScore(clip_model, image_paths, candidates, device)

    # load pacscore clip model
    clip_model_, _ = clip.load(args.clip_model, device=device)
    checkpoint = torch.load(_MODELS[args.clip_model])
    clip_model_.load_state_dict(checkpoint['state_dict'])
    clip_model_.eval()

    # pacscore
    _, pac_scores, candidate_feats_, len_candidates = PACScore(clip_model_, image_paths, candidates, device, w=2.0)

    if args.references_json:
        # refclipscore
        _, per_instance_text_text = RefCLIPScore(clip_model, references, candidate_feats, device)
        refclip_scores = 2 * clip_scores * per_instance_text_text / (clip_scores + per_instance_text_text) # F-score
    
        # refpacscore
        _, per_instance_text_text = RefPACScore(clip_model_, references, candidate_feats_, device, torch.tensor(len_candidates))
        refpac_scores = 2 * pac_scores * per_instance_text_text / (pac_scores + per_instance_text_text)

        scores = {
            image_id: {
                'CLIPScore': float(clipscore), 
                'RefCLIPScore': float(refclipscore), 
                'PACScore': float(pacscore), 
                'RefPACScore': float(refpacscore),
                'SentenceScore': float(senscore)
                }
                  for image_id, clipscore, refclipscore, pacscore, refpacscore, senscore in
                  zip(image_ids, clip_scores, refclip_scores, pac_scores, refpac_scores, sen_scores)} 

    else:
        scores = {image_id: {'CLIPScore': float(clipscore), 'PACScore': float(pacscore), 'SentenceScore': float(senscore)}
                  for image_id, clipscore, pacscore, senscore in
                  zip(image_ids, clip_scores, pac_scores, sen_scores)}
        print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))
        print('PACScore: {:.4f}'.format(np.mean([s['PACScore'] for s in scores.values()])))
        print('SentenceScore: {:.4f}'.format(np.mean([s['SentenceScore'] for s in scores.values()])))

    if args.references_json:
        other_metrics = get_all_metrics(references, candidates)
        res_dir = os.path.dirname(args.candidates_json)
        res_path = os.path.join(res_dir, 'results_.txt')
        with open(res_path, 'a+') as f:
            f.write('\nEvaluation time: ' + str(datetime.datetime.now()) + '\n')
            f.write('References path: ' + args.references_json + '\n')
            f.write('Candidates path: ' + args.candidates_json + '\n\n')
            for k, v in other_metrics.items():
                if k == 'bleu':
                    for bidx, sc in enumerate(v):
                        # print('BLEU-{}: {:.4f}'.format(bidx+1, sc))
                        line = 'BLEU-{}: {:.4f}'.format(bidx+1, sc)
                        print(line)
                        f.write(line + '\n')

                else:
                    # print('{}: {:.4f}'.format(k.upper(), v))
                    line = '{}: {:.4f}'.format(k.upper(), v)
                    print(line)
                    f.write(line + '\n')
            # print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))
            # print('RefCLIPScore: {:.4f}'.format(np.mean([s['RefCLIPScore'] for s in scores.values()])))
            # print('PACScore: {:.4f}'.format(np.mean([s['PACScore'] for s in scores.values()])))
            # print('RefPACScore: {:.4f}'.format(np.mean([s['RefPACScore'] for s in scores.values()])))
            # print('SentenceScore: {:.4f}'.format(np.mean([s['SentenceScore'] for s in scores.values()])))
            line = 'CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()]))
            print(line)
            f.write(line + '\n')
            line = 'RefCLIPScore: {:.4f}'.format(np.mean([s['RefCLIPScore'] for s in scores.values()]))
            print(line)
            f.write(line + '\n')
            line = 'PACScore: {:.4f}'.format(np.mean([s['PACScore'] for s in scores.values()]))
            print(line)
            f.write(line + '\n')
            line = 'RefPACScore: {:.4f}'.format(np.mean([s['RefPACScore'] for s in scores.values()]))
            print(line)
            f.write(line + '\n')
            line = 'SentenceScore: {:.4f}'.format(np.mean([s['SentenceScore'] for s in scores.values()]))
            print(line)
            f.write(line + '\n')

    if args.save_per_instance:
        with open(args.save_per_instance, 'w') as f:
            f.write(json.dumps(scores))

if __name__ == '__main__':
    main()