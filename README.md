# PyTorch-YOLOv3
Minimal implementation of YOLOv3 in PyTorch.

## Table of Contents
- [PyTorch-YOLOv3](#pytorch-yolov3)
  * [Table of Contents](#table-of-contents)
  * [Paper](#paper)
  * [Installation](#installation)
  * [Inference](#inference)
  * [Test](#test)
  * [Train](#train)
  * [Credit](#credit)

## Paper
### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)

## Usage
First copy the configuration:

    $ cp docker/conf.env.dist docker/conf.env
    
Modify the environment variable to point to the right directories on your local machine.
You can then build and run the services:

    $ docker-compose build
    $ docker-compose run yolov3 --help

##### Download pretrained weights
    $ docker-compose run download_weights

##### Download COCO
    $ docker-compose run download_coco

## Inference
Uses pretrained weights to make predictions on images. Below table displays the inference times when using as inputs images scaled to 256x256. The ResNet backbone measurements are taken from the YOLOv3 paper. The Darknet-53 measurement marked shows the inference time of this implementation on my 1080ti card.

| Backbone                | GPU      | FPS      |
| ----------------------- |:--------:|:--------:|
| ResNet-101              | Titan X  | 53       |
| ResNet-152              | Titan X  | 37       |
| Darknet-53 (paper)      | Titan X  | 76       |
| Darknet-53 (this impl.) | 1080ti   | 74       |

```
Usage: yolov3 predict [OPTIONS] [INPUT_PATH] [OUTPUT_PATH] [CONFIG_PATH]
                      [WEIGHTS_PATH] [CLASS_PATH]

Options:
  --conf_thres FLOAT    object confidence threshold
  --nms_thres FLOAT     iou thresshold for non-maximum suppression
  --batch_size INTEGER  size of the batches
  --n_cpu INTEGER       number of cpu threads to use during batch generation
  --img_size INTEGER    size of each image dimension
  --use_cuda            whether to use cuda if available
  --plot                output images with bounding boxes
  --help                Show this message and exit.
```

<p align="center"><img src="assets/giraffe.png" width="480"\></p>
<p align="center"><img src="assets/dog.png" width="480"\></p>
<p align="center"><img src="assets/traffic.png" width="480"\></p>
<p align="center"><img src="assets/messi.png" width="480"\></p>

## Test
Evaluates the model on COCO test.

| Model                   | mAP (min. 50 IoU) |
| ----------------------- |:----------------:|
| YOLOv3 (paper)          | 57.9             |
| YOLOv3 (this impl.)     | 58.2             |


```
Usage: yolov3 test [OPTIONS] [MODEL_CONFIG_PATH] [DATA_CONFIG_PATH]
                   [WEIGHTS_PATH]

Options:
  --batch_size INTEGER  size of each image batch
  --iou_thres FLOAT     Iou thresshold for non-maximum suppression
  --conf_thres FLOAT    object confidence threshold
  --nms_thres FLOAT     Nms thresshold for non-maximum suppression
  --n_cpu INTEGER       number of cpu threads to use during batch generation
  --img_size INTEGER    size of each image dimension
  --use_cuda            whether to use cuda if available
  --verbose             Output training information
  --help                Show this message and exit.
```

## Train
Model does not converge yet during training. Data augmentation as well as additional training tricks remains to be implemented. PRs are welcomed!
```
Usage: yolov3 train [OPTIONS] [INPUT_PATH] [MODEL_CONFIG_PATH]
                    [DATA_CONFIG_PATH] [WEIGHTS_PATH] [CHECKPOINT_DIR]

Options:
  --nb_epochs INTEGER            number of epochs
  --batch_size INTEGER           size of each image batch
  --conf_thres FLOAT             object confidence threshold
  --nms_thres FLOAT              iou thresshold for non-maximum suppression
  --n_cpu INTEGER                number of cpu threads to use during batch
                                 generation
  --img_size INTEGER             size of each image dimension
  --checkpoint_interval INTEGER  interval between saving model weights
  --use_cuda                     whether to use cuda if available
  --verbose                      Output training information
  --help                         Show this message and exit.
```

## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
