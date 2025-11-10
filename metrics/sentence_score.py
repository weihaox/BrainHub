#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   sentence_score.py
@Time    :   2025/03/22 20:42:35
@Author  :   Weihao Xia 
@Version :   1.0
@Desc    :   
usage:
'''

import collections
import numpy as np
import tqdm
import torch
from metrics import CLIPCapDataset
from sentence_transformers import util

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def SentenceScore(model, references, candidates, device):
    '''
    compute the sentence similarity.
    '''
    len_candidates = [len(c.split()) for c in candidates]
    candidates = model.encode(candidates, device=device)
    
    len_references = []
    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        len_r = [len(r.split()) for r in refs]
        len_references.append(len_r)
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = model.encode(flattened_refs, device=device)
    
    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, (cand, l_ref, l_cand) in enumerate(zip(candidates, len_references, len_candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose()) # array([ 0.08097363,  0.05734243, -0.00745275,  0.04417423, -0.03602952], dtype=float32)
        # all_sims = util.pytorch_cos_sim(cand, cur_refs).cpu().numpy() # array([[ 0.08097364,  0.05734244, -0.00745277,  0.04417423, -0.03602953]], dtype=float32)
        per.append(np.max(all_sims))

    return np.mean(per), per