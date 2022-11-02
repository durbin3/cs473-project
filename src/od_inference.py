import os
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

# Enable GPU dynamic memory allocation
for gpu in tf.config.experimental.list_physical_devices('GPU') : tf.config.experimental.set_memory_growth(gpu, True)

LABEL_MAP = 'dataset/label_map.pbtxt'
PATH_TO_SAVED_MODEL = 'models/exported_models/object_detection/saved_model'
IMAGE_SIZE = (1024,1024)

def object_detection(image_path):
    if not os.path.exists('./out'): os.makedirs('./out')
    image = load_image(image_path)
    image_np = np.array(image)[:,:,:3]
    image_tensor,scale_factor = preprocess_image(image)
    model = load_model(PATH_TO_SAVED_MODEL)
    print(f'Detecting objects in {image_path[-7:]}', end='')
    detections = detect_objects(image_tensor,model)
    detections['detection_boxes'] = detections['detection_boxes'] * scale_factor
    visualize_detections(detections,image_np,'./out/' + image_path[-7:])
    print('\tDone')


def load_model(path):
    print('Loading model...', end='')
    start_time = time.time()
    if not os.path.exists(path):
        print("\nError, no saved model found")
        exit()
    model = tf.saved_model.load(path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return model

def load_image(image_path):
    return Image.open(image_path)

def preprocess_image(image):
    square = expand_to_square(image)
    resized_image = square.resize(IMAGE_SIZE,resample=Image.Resampling.LANCZOS)
    scale_factor = square.size[0]/IMAGE_SIZE[0]
    image_np = np.array(resized_image)[:,:,:3]
    input_tensor = tf.convert_to_tensor(image_np)   # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = input_tensor[tf.newaxis, ...]    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    return input_tensor,scale_factor

def detect_objects(image_tensor,model):
    detections = model(image_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)  # detection_classes should be ints.
    return detections

def visualize_detections(detections,image,out_path):
    image_np_with_detections = image.copy()
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP, use_display_name=True)
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.savefig('./out/'+out_path)
    plt.show()

def expand_to_square(image):
    width, height = image.size
    if width == height:
        return image
    elif width > height:
        result = Image.new(image.mode, (width, width), (255,255,255))
        result.paste(image)
        return result
    else:
        result = Image.new(image.mode, (height, height), (255,255,255))
        result.paste(image)
        return result
