# BrainHub: Multimodal Brain Understanding Benchmark

## Updates
- [2024/07/01] [UMBRAE](https://github.com/weihaox/UMBRAE) is accepted to ECCV 2024.
- [2024/05/18] Update Leaderboard results.
- [2024/04/11] The brainhub benchmark has been released.

## Motivation

Unlike texts, images, or audio, whose contents are intuitively aligned with human perception and judgment, we lack sufficient knowledge of the information contained in captured brain responses, as they are not directly interpretable or interoperable to humans. We could translate the brain's responses into other understandable modalities as an indirect method of ascertaining its ability to describe, recognize, and localize instances, as well as discern spatial relationships among multiple exemplars. These abilities are important for brain-machine interfaces and other brain-related research. Therefore, we construct BrainHub, a brain understanding benchmark, based on [NSD](https://naturalscenesdataset.org/) and [COCO](https://cocodataset.org). 

## Tasks and Metrics

The objectives are categorized into concept recognition and spatial localization, including: 

- brain captioning, which is to generate textual descriptions summarizing the primary content of a given brain response. To evaluate the quality of generated captions, we use five [standard metrics](https://github.com/tylin/coco-caption), BLEU, METEOR, ROUGE, CIDEr, and SPICE, in addition to [CLIP-based scores](https://github.com/jmhessel/clipscore), CLIP-S and RefCLIP-S, for comprehensive multimodal evaluation. **Update Nov 2025**: add support for [PAC-Score](https://github.com/aimagelab/pacscore), [PAC-Score++](https://github.com/aimagelab/pacscore?tab=readme-ov-file#compute-pac-s-1), and [SentenceS](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

- brain grounding, which is the counterpart of visual grounding and seeks to recover spatial locations from brain responses by inferring coordinates. Given that identified classes might be named differently, or simply absent from ground truth labels, we evaluate boundingboxes through REC, using accuracy and IoU as metrics.

## Evaluation

There are 982 test images, 80 classes, 4,913 captions, and 5,829 boundingboxes. For grounding evaluation, we further group the 80 classes of COCO into four salience categories according to their salience in images: Salient (S), Salient Creatures (SC), Salient Objects (SO), and Inconspicuous (I). The illustration shows the statistics and mapping of our categories, w.r.t. COCO classes.

We provide the processed [text](https://github.com/weihaox/brainhub/caption) and [boundingbox](https://github.com/weihaox/brainhub/bbox) groundtruth. The demo evaluation script is provided [here](https://github.com/weihaox/brainhub/run.sh). If you would like to evaluate your produced results, please modify the result path accordingly.

We also provide baseline results associated with BrainHub, including the captioning results from [SDRecon](https://github.com/yu-takagi/StableDiffusionReconstruction), [BrainCap](https://arxiv.org/abs/2305.11560), [OneLLM](https://onellm.csuhan.com/), [MEVOX](https://cvpr25-advml.github.io/short_paper/21_Multi_Task_Vision_Experts_.pdf), [BrainDEC](https://github.com/Yf-Hma/brain_decode), and [MindEye2](https://medarc-ai.github.io/mindeye2), as well as the captioning and grounding results from [UMBRAE](https://weihaox.github.io/UMBRAE/). 

For contributing, please (a) update the leaderboard and (b) upload the results to the desired path with the required file name, such as `caption/comparison/umbrae/cross_sub/sub01_decoded_caption.json`.

## Leaderboard

### Captioning

'UMBRAE-S1' refers to model trained with S1 only, while 'UMBRAE' denotes the model with cross-subject training. UMBRAE is the **first zero-shot multimodal** brain decoding method.

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS | PACScore | RefPACS | SentenceS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|----------|---------|-----------|
| UMBRAE    | S1   | 59.44 | 19.03 | 19.45  | 43.71 | 61.06 | 12.79 | 67.78 | 73.54    | 70.76    | 76.75   | 57.75     |
| UMBRAE-S1 | S1   | 57.74 | 17.79 | 18.28  | 42.22 | 52.53 | 12.08 | 66.59 | 72.41    | 69.97    | 75.90   | 55.53     |
| MEVOX     | S1   | 58.56 | 20.11 | 19.20  | 44.47 | 54.37 | 11.05 | 64.23 | 70.36    | 67.51    | 73.85   | 53.07     |
| MindEye2  | S1   | 54.82 | 18.16 | 17.54  | 43.79 | 55.70 | 10.97 | 67.54 | 73.73    | 70.58    | 76.86   | 56.85     |
| BrainCap  | S1   | 55.96 | 14.51 | 16.68  | 40.69 | 41.30 | 9.06  | 64.31 | 69.90    | 67.69    | 73.50   | 50.57     |
| OneLLM    | S1   | 47.04 | 9.51  | 13.55  | 35.05 | 22.99 | 6.26  | 54.80 | 61.28    | 58.39    | 65.24   | 37.73     |
| SDRecon   | S1   | 36.21 | 3.43  | 10.03  | 25.13 | 13.83 | 5.02  | 61.07 | 66.36    | 65.24    | 70.53   | 39.97     |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS | PACScore | RefPACS | SentenceS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|----------|---------|-----------|
| UMBRAE    | S2   | 59.37 | 18.41 | 19.17  | 43.86 | 55.93 | 12.08 | 66.46 | 72.36    | 69.56    | 75.68   | 56.21     |
| UMBRAE-S2 | S2   | 56.68 | 17.15 | 18.18  | 41.77 | 51.91 | 11.74 | 66.12 | 71.90    | 69.52    | 75.52   | 54.98     |
| BrainCap  | S2   | 53.80 | 13.03 | 15.90  | 39.96 | 35.60 | 8.47  | 62.48 | 68.19    | 66.14    | 72.07   | 48.02     |
| SDRecon   | S2   | 34.71 | 3.02  | 9.60   | 24.22 | 13.38 | 4.58  | 59.52 | 65.30    | 64.52    | 69.87   | 38.63     |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS | PACScore | RefPACS | SentenceS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|----------|---------|-----------|
| UMBRAE    | S5   | 60.36 | 19.03 | 20.04  | 44.81 | 61.32 | 13.19 | 68.39 | 74.11    | 71.64    | 77.53   | 59.14     |
| UMBRAE-S5 | S5   | 59.37 | 18.37 | 19.13  | 43.18 | 58.55 | 12.85 | 67.70 | 73.47    | 71.06    | 76.98   | 58.02     |
| BrainCap  | S5   | 55.28 | 14.62 | 16.45  | 40.87 | 41.05 | 9.24  | 63.89 | 69.64    | 67.55    | 73.43   | 50.34     |
| SDRecon   | S5   | 34.96 | 3.49  | 9.93   | 24.77 | 13.85 | 5.19  | 60.83 | 66.30    | 65.90    | 70.99   | 40.53     |

| Method    | Eval | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS | PACScore | RefPACS | SentenceS |
|-----------|------|-------|-------|--------|-------|-------|-------|-------|----------|----------|---------|-----------|
| UMBRAE    | S7   | 57.20 | 17.13 | 18.29  | 42.16 | 52.73 | 11.63 | 65.90 | 71.83    | 68.98    | 75.13   | 54.90     |
| UMBRAE-S7 | S7   | 55.85 | 14.99 | 17.38  | 40.68 | 46.97 | 10.91 | 65.01 | 70.91    | 68.22    | 74.38   | 52.82     |
| BrainCap  | S7   | 54.25 | 14.00 | 15.94  | 40.02 | 37.49 | 8.57  | 62.52 | 68.48    | 66.03    | 72.10   | 47.70     |
| SDRecon   | S7   | 34.99 | 3.26  | 9.54   | 24.33 | 13.01 | 4.74  | 58.68 | 64.59    | 64.05    | 69.47   | 37.39     |

### Grounding

| Method    | Eval | acc@0.5 (A) | IoU (A) | acc@0.5 (S) | IoU (S) | acc@0.5 (I) | IoU (I) |
|-----------|------|-------------|---------|-------------|---------|-------------|---------|
| UMBRAE    | S1   | 18.93       | 21.28   | 30.23       | 30.18   | 4.83        | 10.18   |
| UMBRAE-S1 | S1   | 13.72       | 17.56   | 21.52       | 25.14   | 4.00        | 8.08    |
| UMBRAE    | S2   | 18.27       | 20.77   | 28.22       | 29.19   | 5.86        | 1025    |
| UMBRAE-S2 | S2   | 15.21       | 18.68   | 23.60       | 26.59   | 4.74        | 8.81    |
| UMBRAE    | S5   | 18.19       | 20.85   | 28.74       | 30.02   | 5.02        | 9.41    |
| UMBRAE-S5 | S5   | 14.72       | 18.45   | 22.93       | 26.34   | 4.46        | 8.60    |
| UMBRAE    | S7   | 16.74       | 19.63   | 25.69       | 27.90   | 5.58        | 9.31    |
| UMBRAE-S7 | S7   | 13.60       | 17.83   | 21.07       | 25.19   | 4.28        | 8.64    |

## Citation

```bibtex
@inproceedings{xia2024umbrae,
  author    = {Xia, Weihao and de Charette, Raoul and Öztireli, Cengiz and Xue, Jing-Hao},
  title     = {UMBRAE: Unified Multimodal Brain Decoding},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```