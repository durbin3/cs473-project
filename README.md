python version: 3.9.13

to setup:

```sh
python -m venv venv
source scripts/setup
```
Your file tree must look like below, with the raw images placed in the `dataset/raw_images` folder.

```
CS473-Project
├─ dataset
|   ├─ images
|   |   └─ raw_images
|   ├─ labels
|   ├─ label_map.pbtxt
├─ scripts
|   └─ ...
├─ venv
└─ ...
```
All else fails follow this: [Setup tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

## Premade Scripts
Here's a list of scripts and their descriptions for what they do, in approximate order of when you will probably run them:
- `scripts/setup`: used to install most packages
- `scripts/protobuf_fix`: steals a build file from a later version of protobuf that is used for the object detection api
- `scripts/gen_dataset.sh`: combines the xml labels and resized images and generates a train.record file and a test.record file for inputs to the model training.
- `scripts/train_object_detection`: runs the object detection training model
- `scripts/export_object_detection`: takes the most recent checkpoint of the object detection model and exports it to a finalized prediction model.

## Image Processing
To preprocess the images, run `python scripts/image_resizing.py process 1024`
This should resize all the images to be 1024x1024 pixels and place them in `dataset/images/resized_images`

## Object detection
Object detection requires a specific tensorflow package that must be installed manually (unfortunately). To do so, follow [this guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation)

If an import error for the object detection api comes up regarding builder.py, you may need to run `source scripts/protobuf_fix`

If an error pops up relating to tensorflow not being able to find {CUDA_DIR}, you may need to go to the directory where CUDA is, and copy-paste the `nvvm` folder into this project's source directory.

## Model Installation
Here's a list of all the models used
- [Object detection](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8.tar.gz)

After downloading, extract the models and place them in the `models/pretrained_models` folder, which you may need to create.
