# Assembly staet detection task
## Installation 

First pip install the ultralytics package including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/). Please check the ultralytics [repository](https://github.com/ultralytics/ultralytics/tree/main) for more detailed instructions.
```
$ pip install ultralytics
```
Then download the scripts and install additional requirements.
```
$ git clone https://github.com/TimSchoonbeek/IndustReal
$ pip install pylabel
```
Please download the pre-trained COCO model from the ultralytics [repository](https://github.com/ultralytics/ultralytics). The VOC2012 dataset used for mixup data augmentation can be found on [this](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) page.

## Usage
This repo is mostly adpot from the our previous work. So the usage of script to train the object detection stream model can be found in this [repo](https://github.com/TimSchoonbeek/IndustReal/tree/main/ASD). 