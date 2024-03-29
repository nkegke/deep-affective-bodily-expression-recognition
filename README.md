# Deep Affective Bodily Expression Recognition

<img src="https://github.com/nkegke/files/blob/main/thesis/baseline2.png" alt="tsn" style="width: 55vw;"/>
<p align="center">
TSN-Based Visual Emotion Recognition Model
</p>

## Abstract
> The COVID-19 pandemic has undoubtedly changed the standards and affected all aspects of our lives, especially social communication. It has forced people to extensively wear medical face masks, in order to prevent transmission. This face occlusion can strongly irritate emotional reading from the face and urges us to incorporate the whole body as an emotional cue. In this paper, we conduct insightful studies about the effect of face occlusion on emotion recognition performance, and showcase the superiority of full body input over the plain masked face. We utilize a deep learning model based on the Temporal Segment Network framework, and aspire to fully overcome the face mask consequences. Although facial and bodily features can be learned from a single input, this may lead to irrelevant information confusion. By processing those features separately and fusing their prediction scores, we are more effectively taking advantage of both modalities. This framework also naturally supports temporal modeling, by mingling information among neighboring frames. In combination, these techniques form an effective system capable of tackling emotion recognition difficulties, caused by safety protocols applied in crucial areas.

## Prerequisites

* Linux
* CUDA 11.2
* numpy 1.21.5
* matplotlib 3.5.1
* torch 1.11.0
* torchvision 0.12.0
* PIL 9.0.1
* opencv-python 4.5.5
* pandas 1.4.2
* sklearn 1.0.2
* pytorch_grad_cam
* gdown

**Dataset PATHs:** 
* [EmoReact/dataset.py](EmoReact/dataset.py): line 228 (mp4), 113 (OpenFace), 118 (holistic)
* [EmoReact/dataset.py](EmoReact/dataset.py): lines 237-241 (extracted frames using [vid2img](tools/vid2img_emoreact_mask.py))
* [extract_hol.py](extract_hol.py), [prepare_hol_txt.py](prepare_hol_txt.py): holistic landmark extraction


**Dataset CSVs:** [EmoReact/{train,val,test}.csv](EmoReact/train.csv)

## How to run
```
python train_EmoReact.py [--input {face,body,fullbody,fusion}]
                         [--mask]
                         [--num_segments {1,3,5,10}]
                         [--arch {resnet50,mobilenet_v2}]
                         [--shift] [--shift_div {4,8}]
                         --config EmoReact/config.json
```

Face |  Full Body | Fusion (Face + Body)
:-------:|:----------:|:----------:
<img src="https://github.com/nkegke/files/blob/main/thesis/face_mask.png" alt="face" style="width:151px; height:200px"/> | <img src="https://github.com/nkegke/files/blob/main/thesis/body_mask.png" alt="face" style="width:305px; height:200px"/> | <img src="https://github.com/nkegke/files/blob/main/thesis/fusion.png" alt="face" style="width:370px; height:200px"/>


## Visual Explanation (using [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam))

Excitement | Frustration
:---------:|:----------:
<img src="https://github.com/nkegke/files/blob/main/thesis/ef1.png" alt="face" style="width: 8vw;"/> <img src="https://github.com/nkegke/files/blob/main/thesis/ef2.png" alt="face" style="width: 8vw;"/> <img src="https://github.com/nkegke/files/blob/main/thesis/ef3.png" alt="face" style="width: 8vw;"/> | <img src="https://github.com/nkegke/files/blob/main/thesis/ff1.png" alt="face" style="width: 8vw;"/> <img src="https://github.com/nkegke/files/blob/main/thesis/ff2.png" alt="face" style="width: 8vw;"/> <img src="https://github.com/nkegke/files/blob/main/thesis/ff3.png" alt="face" style="width: 8vw;"/>
<img src="https://github.com/nkegke/files/blob/main/thesis/eb1.png" alt="face" style="width: 8vw;"/> <img src="https://github.com/nkegke/files/blob/main/thesis/eb2.png" alt="face" style="width: 8vw;"/> <img src="https://github.com/nkegke/files/blob/main/thesis/eb3.png" alt="face" style="width: 8vw;"/> | <img src="https://github.com/nkegke/files/blob/main/thesis/fb1.png" alt="face" style="width: 8vw;"/> <img src="https://github.com/nkegke/files/blob/main/thesis/fb2.png" alt="face" style="width: 8vw;"/> <img src="https://github.com/nkegke/files/blob/main/thesis/fb3.png" alt="face" style="width: 8vw;"/>

## Acknowledgements

* [https://github.com/filby89/multimodal-emotion-recognition](https://github.com/filby89/multimodal-emotion-recognition)
* [https://github.com/yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks)
* [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
* [https://github.com/mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module)
