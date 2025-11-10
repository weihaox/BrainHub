#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   clipscore.py
@Time    :   2025/03/22 17:08:50
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   
'''

import numpy as np
from PIL import Image
import collections
import clip
import torch

from metrics import extract_all_captions, extract_all_images

def PACScore(model, images, candidates, device, w=2.0):
    '''
    compute the unreferenced PAC score.
    '''
    len_candidates = [len(c.split()) for c in candidates] 
    if isinstance(images, list):
        # extracting image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
    candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates, len_candidates

def RefPACScore(model, references, candidates, device, len_candidates):
    '''
    compute the RefPAC score, extracting only the reference captions.
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    len_references = []
    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        len_r = [len(r.split()) for r in refs]
        len_references.append(len_r)
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)
    
    candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))
    flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs ** 2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, (cand, l_ref, l_cand) in enumerate(zip(candidates, len_references, len_candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())

        per.append(np.max(all_sims))

    return np.mean(per), per