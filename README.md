# Deep Affective Bodily Expression Recognition

<img src="https://drive.google.com/uc?export=view&id=1OPR42S6ld9sNWo8-VXTU1TULHlIDd_Xh" alt="model" style="width: 55vw;"/>
<p align="center">
TSN-Based Visual Emotion Recognition Model
</p>

## Abstract
> The COVID-19 pandemic has forced people to extensively wear medical face masks, in order to prevent transmission. This face occlusion results in considerable emotion recognition performance drop by models that exploit facial expressions. Therefore, it urges us to incorporate the whole body in the input, as it needs to play a more major role in the task of recognition, despite its complementary nature. Emotional expressions consist of multiple stages spanning over a period of time, which indicates we should not only exploit spatial information from multiple sparsely sampled frames, but also model temporal structure. Although single RGB stream models can learn both face and body features, this may lead to irrelevant information confusion. By processing those features separately and fusing their preliminary prediction scores with a late fusion scheme, we are more effectively taking advantage of both modalities. The [TSN](https://github.com/yjxiong/temporal-segment-networks) architecture can also naturally support temporal modeling, by mingling information among neighboring snippet frames with the [TSM](https://github.com/mit-han-lab/temporal-shift-module) module. Experimental results suggest that spatial structure plays a more important role for an emotional expression, while temporal structure is complementary.

## Prerequisites

* Ubuntu 16.04 LTS
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

**Dataset PATHs:** [EmoReact/dataset.py](EmoReact/dataset.py) line 228 (mp4 + OpenFace) and lines 237-241 (extracted frames using [vid2img](tools/vid2img_emoreact_mask.py))

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
<img src="https://drive.google.com/uc?export=view&id=10ZM8uhDyLX3Y3hzwIGacRc7j1UnYc9xi" alt="face" style="width:151px; height:200px"/>  |   <img src="https://drive.google.com/uc?export=view&id=1gahsNQ_ouKVFXia0Ty0RxwUvqg-ynCpu" alt="fbody" style="width:305px; height:200px"/>  |  <img src="https://drive.google.com/uc?export=view&id=1Zog4VBUH7KCZSFbKc-Qa-kAYVLjyXHtI" alt="fus" style="width:370px; height:200px"/>


## Results

| Model | Input - 3 Segments |  ROC AUC |
|-------|--------------------|----------|
| TSN   | Face               |  0.733   |
| TSN   | Body               |  0.736   |
| TSN   | Full Body          |  0.758   |
| TSM   | Fusion             |  0.768   |
| TSN   | Unmasked Face      |  0.769   |

## Visual Explanation (using [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam))

Excitement | Frustration
:---------:|:----------:
<img src="https://drive.google.com/uc?export=view&id=1fgsMQa6DtB4me-Ayh184ILL4uXCyie65" alt="ef1" style="width: 8vw;"/> <img src="https://drive.google.com/uc?export=view&id=1hkUltmuvqTBx-MTbGY-JxyQoxATTmE1J" alt="ef2" style="width: 8vw;"/> <img src="https://drive.google.com/uc?export=view&id=1VrYHDFYpjna_4IQ746VLThxgGuKiyFdo" alt="ef3" style="width: 8vw;"/> | <img src="https://drive.google.com/uc?export=view&id=1Kfrxi6Cd9baqL4mBKa0KowFg8Uo0VmwS" alt="eb1" style="width: 8vw;"/> <img src="https://drive.google.com/uc?export=view&id=1xujYN3G0ERrcO1Y4HRb-pbeRI2xXj7fw" alt="eb2" style="width: 8vw;"/> <img src="https://drive.google.com/uc?export=view&id=1ht04ONy6IZb0rXn04PjoTxlri0gDOwJV" alt="eb3" style="width: 8vw;"/>
<img src="https://drive.google.com/uc?export=view&id=1AEL03UAVVczkE5xrZSCEYTw75TFkw-LM" alt="ff1" style="width: 8vw;"/> <img src="https://drive.google.com/uc?export=view&id=1ZiFKPm6gfGhS1kVG-vKVxSF5LCFXf4su" alt="ff2" style="width: 8vw;"/> <img src="https://drive.google.com/uc?export=view&id=1-T08W3b2WNEIaxzWSZCJQ_uoNNAYRIbk" alt="ff3" style="width: 8vw;"/> | <img src="https://drive.google.com/uc?export=view&id=1OrsNnN8FbYgCo-lsrK9gskaJjI8sRq1Q" alt="fb1" style="width: 8vw;"/> <img src="https://drive.google.com/uc?export=view&id=1Xx8WqsnrkrY3C7iq9mSDYBKI4xvv0_C0" alt="fb2" style="width: 8vw;"/> <img src="https://drive.google.com/uc?export=view&id=1tXDNNhhG6R0YAzkPa5itHyhNips995Xc" alt="fb3" style="width: 8vw;"/>

## Acknowledgements

* [https://github.com/filby89/multimodal-emotion-recognition](https://github.com/filby89/multimodal-emotion-recognition)
* [https://github.com/yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks)
* [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
* [https://github.com/mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module)
