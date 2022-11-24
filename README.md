## For TAs
To run object detection, after setup, run `python src/od_inference.py [model_path] [image_path]`
Note that `model_path` must be the path to the saved model directory, e.g `models/exported_models/centernet512_7000_no_aug/saved_model/`.
A valid command for running should be: `python src/od_inference.py models/exported_models/centernet512_7000_no_aug/saved_model/ dataset/images/raw_images/120.png`
The output will be in `./out` and will contain an image of the object detection (if the object_detection package is installed) and a text file that contains the output list.

To run OCR run `python src/ocr_main.py -r ./out -o ./out/ocr -i [images folder path]`
This will run OCR on all of the images and the detected objects found in the `./out` folder.

## Setup
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
To preprocess the images, run `python src/image_resizing.py process 1024`
This should resize all the images to be 1024x1024 pixels and place them in `dataset/images/resized_images`

## Object detection
Object detection requires a specific tensorflow package that must be installed manually (unfortunately). To do so, follow [this guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation)

If an import error for the object detection api comes up regarding builder.py, you may need to run `source scripts/protobuf_fix`

If an error pops up relating to tensorflow not being able to find {CUDA_DIR}, you may need to go to the directory where CUDA is, and copy-paste the `nvvm` folder into this project's source directory.

The basic running steps for training are:
- `python src/image_resizing.py resize [size]` to resize the images and labels for the models input size
- `source scripts/gen_dataset.sh` to generate the test/train datasets
- `source scripts/train_object_detection [model_name]` to train the model
- `source scripts/export_object_detection [model_name]` to export the most recent checkpoint to be a model
- `python src/od_inference.py [model_name] [path] [image_size]` to run object detection on an image

To run object detection for inference:
- `python src/od_inference.py [model_path] [image_path]`

## Model Installation
Here's a list of the models used
- [centernet512_7000_no_aug](https://drive.google.com/file/d/16LcVmtmh_rJ3eGczJFKfRi-SAv-tpYV0/view?usp=drivesdk) (Link to Google Doc with exported model)

After downloading, extract the models and place them in the `models/pretrained_models` folder, which you may need to create.

## OCR

Run OCR scripts based on Object Detection output saved in `out/*.txt`

Running step:
- `python src/ocr_main.py -r ./out -o ./out/ocr -i ./dataset/images/raw_images` to extract texts and save it in `out/ocr`

## Base Clustering
Run Baseline Clustering based on OCR results saved in `out/ocr/`
Running step:
- `python ./stage2/baseClustering.py -r ./out/ocr -k [number of clusters] -o [output result path]
