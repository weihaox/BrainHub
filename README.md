# BrainHub: Multimodal Brain Understanding Benchmark

## Updates
- [2024/04/11] The brainhub benchmark has been released.

## Motivation

Unlike texts, images, or audio, whose contents are intuitively aligned with human perception and judgment, we lack sufficient knowledge of the information contained in captured brain responses, as they are not directly interpretable or interoperable to humans. We could translate the brain's responses into other understandable modalities as an indirect method of ascertaining its ability to describe, recognize, and localize instances, as well as discern spatial relationships among multiple exemplars. These abilities are important for brain-machine interfaces and other brain-related research. Therefore, we construct BrainHub, a brain understanding benchmark, based on [NSD](https://naturalscenesdataset.org/) and [COCO](https://cocodataset.org). 

## Tasks and Metrics

These objectives can generally be categorized into concept recognition and spatial localization, encompassing two tasks: 

- brain captioning, which is to generate textual descriptions summarizing the primary content of a given brain response. To evaluate the quality of generated captions, we use five [standard metrics](https://github.com/tylin/coco-caption), BLEU, METEOR, ROUGE, CIDEr, and SPICE, in addition to [CLIP-based scores](https://github.com/jmhessel/clipscore), CLIP-S and RefCLIP-S.

- brain grounding, which is the counterpart of visual grounding and seeks to recover spatial locations from brain responses by inferring coordinates. Given that identified classes might be named differently, or simply absent from ground truth labels, we evaluate boundingboxes through REC, using accuracy and IoU as metrics.

## Evaluation

There are 982 test images, 80 classes, 4,913 captions, and 5,829 boundingboxes. For grounding evaluation, we further group the 80 classes of COCO into 4 salience categories according to their salience in images: Salient (S), Salient Creatures (SC), Salient Objects (SO), and Inconspicuous (I). The illustration shows the statistics and mapping of our categories, w.r.t. COCO classes.

We provide the processed [text](https://github.com/weihaox/brainhub/caption) and [boundingbox](https://github.com/weihaox/brainhub/bbox) groundtruth. The demo evaluation script is provided [here](https://github.com/weihaox/brainhub/run.sh). If you would like to evaluate your produced results, please modify the result path accordingly.

We also provide baseline results associated with BrainHub, including the captioning results from [SDRecon](https://github.com/yu-takagi/StableDiffusionReconstruction), [BrainCap](https://arxiv.org/abs/2305.11560), and [OneLLM](https://onellm.csuhan.com/), as well as the captioning and grounding results from [UMBRAE](https://weihaox.github.io/UMBRAE/). 

## Leaderboard

### Captioning

This is the quantitative comparison for subject 1 (S1). For more results on other subjects, refer to the [UMBRAE](https://weihaox.github.io/UMBRAE/) paper. 'UMBRAE-S1' refers to model trained with a single subject (S1 here) only, while 'UMBRAE' denotes the model with cross-subject training. 

| Method    | BLEU1 | BLEU4 | METEOR | ROUGE | CIDEr | SPICE | CLIPS | RefCLIPS |
|-----------|-------|-------|--------|-------|-------|-------|-------|----------|
| UMBRAE    | 57.84 | 17.17 | 18.70  | 42.14 | 53.87 | 12.27 | 66.10 | 72.33    |
| UMBRAE-S1 | 57.63 | 16.76 | 18.41  | 42.15 | 51.93 | 11.83 | 66.44 | 72.12    |
| BrainCap  | 55.96 | 14.51 | 16.68  | 40.69 | 41.30 | 9.06  | 64.31 | 69.90    |
| OneLLM    | 47.04 | 9.51  | 13.55  | 35.05 | 22.99 | 6.26  | 54.80 | 61.28    |
| SDRecon   | 36.21 | 3.43  | 10.03  | 25.13 | 13.83 | 5.02  | 61.07 | 66.36    |

### Grounding

| Method    | Eval | acc@0.5 (A) | IoU (A) | acc@0.5 (S) | IoU (S) | acc@0.5 (I) | IoU (I) |
|-----------|------|-------------|---------|-------------|---------|-------------|---------|
| UMBRAE-S1 | S1   | 13.72       | 17.56   | 21.52       | 25.14   | 4.00        | 8.08    |
| UMBRAE    | S1   | 15.75       | 19.37   | 24.42       | 27.46   | 4.93        | 9.26    |
| UMBRAE-S2 | S2   | 15.21       | 18.68   | 23.60       | 26.59   | 4.74        | 8.81    |
| UMBRAE    | S2   | 15.30       | 19.09   | 23.60       | 26.96   | 4.93        | 9.27    |
| UMBRAE-S5 | S5   | 14.72       | 18.45   | 22.93       | 26.34   | 4.46        | 8.60    |
| UMBRAE    | S5   | 15.54       | 18.81   | 24.27       | 26.81   | 4.65        | 8.82    |
| UMBRAE-S7 | S7   | 13.60       | 17.83   | 21.07       | 25.19   | 4.28        | 8.64    |
| UMBRAE    | S7   | 14.30       | 18.08   | 22.04       | 25.74   | 4.65        | 8.52    |

## Citation

```bibtex
@article{xia2024umbrae,
  author    = {Xia, Weihao and de Charette, Raoul and Öztireli, Cengiz and Xue, Jing-Hao},
  title     = {UMBRAE: Unified Multimodal Decoding of Brain Signals},
  journal   = {arxiv preprint:arxiv 2404.07202},
  year      = {2024},
}
```