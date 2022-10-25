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

If an import error for the object detection api comes up regarding builder.py, you may need to run `source scripts/protobuf_fix`

## Image Processing
To preprocess the images, run `python scripts/image_resizing.py`
This should resize all the images to be 1000x1000 pixels and place them in `dataset/images/resized_images`
## Object detection
To construct the dataset, you must first label the image set by running `labelImg dataset/images/resized_images` and setting the save_dir to `dataset/labels`. If the label map is not created and located at `dataset/label_map.pbtxt` it must be created.

Then you must merge the set of labels and their corresponding images into `dataset/images/labeled_images`.
Then run the following command to partition the datset into train-test.

```sh
python scripts/partition_dataset.py -i dataset/images/labeled_images/ -o dataset -r .3 -x
```

Afterwards, tf records will need to be created, so run

```sh
python scripts/generate_tfrecord.py -x dataset/train -l dataset/label_map.pbtxt -o dataset/train.record
python scripts/generate_tfrecord.py -x dataset/test -l dataset/label_map.pbtxt -o dataset/test.record
```