import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

# Enable GPU dynamic memory allocation
for gpu in tf.config.experimental.list_physical_devices('GPU') : tf.config.experimental.set_memory_growth(gpu, True)

LABEL_MAP = 'dataset/label_map.pbtxt'
LABELS = ['entity','weak_entity','rel','ident_rel','rel_attr','many','one']
THRESHOLD = .45
def object_detection(image_path,size,model_path):
    if not os.path.exists('./out'): os.makedirs('./out')
    image = load_image(image_path)
    image_np = np.array(image)[:,:,:3]
    image_tensor,scale_factor,small_np = preprocess_image(image,(size,size))
    model = load_model(model_path)
    
    print(f'Detecting objects in {image_path[-7:]}', end='')
    detections = detect_objects(image_tensor,model,size)
    num_detections = len(detections['detection_scores'])
    objects = []
    for i in range(num_detections):
        if detections['detection_scores'][i] > THRESHOLD:
            obj_class_num = detections['detection_classes'][i]
            obj_class = LABELS[obj_class_num-1]
            loc = detections['detection_boxes'][i].tolist()
            objects.append([obj_class,loc])
    print("\nObjects Found: ", objects)

    # Save objects in image into file.
    out_filename = './out/' + image_path[-7:-4] + '.txt' # ./out/001.txt
    print("\nWriting objects to: ", out_filename)
    with open(out_filename, 'w') as out_file:
      out_file.write(str(objects))

    # visualize_detections(detections,small_np,'./out/raw_' + image_path[-7:])
    detections['detection_boxes'] = detections['detection_boxes'] * scale_factor
    visualize_detections(detections,image_np,'./out/' + image_path[-7:])
    print('\tDone')
    return objects

def load_model(path):
    print('Loading model...', end='')
    start_time = time.time()
    if not os.path.exists(path):
        print("\nError, no saved model found")
        exit()
    model = tf.saved_model.load(path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Done! Took {elapsed_time:.2f} seconds')
    return model

def load_image(image_path):
    img = Image.open(image_path)
    print("Loaded image of size: ", img.size)
    return img

def preprocess_image(image,size=(1024,1024)):
    square = expand_to_square(image)
    resized_image = square.resize(size,resample=Image.Resampling.LANCZOS)
    scale_factor = square.size[0]/size[0]
    print(f"Scale Factor: {scale_factor:.3f}")
    image_np = np.array(resized_image)[:,:,:3]
    input_tensor = tf.convert_to_tensor(image_np)   # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = input_tensor[tf.newaxis, ...]    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    return input_tensor,scale_factor,image_np

def detect_objects(image_tensor,model,size):
    detections = model(image_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)  # detection_classes should be ints.
    detections['detection_boxes'] = detections['detection_boxes'] * size
    return detections

def visualize_detections(detections,image,out_path):
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP, use_display_name=True)
    image_np_with_detections = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=False,
            max_boxes_to_draw=200,
            min_score_thresh=.45,
            agnostic_mode=False)

    plt.figure(dpi=400)
    plt.imshow(image_np_with_detections)
    plt.savefig(out_path)
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

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Need to include a model name and a file path to an image")
        exit()

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) == 4 else 512

    object_detection(image_path,size,model_path)