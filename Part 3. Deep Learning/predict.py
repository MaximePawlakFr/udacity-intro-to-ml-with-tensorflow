import argparse
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
import numpy as np
import json

import warnings
warnings.filterwarnings("ignore")
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
   
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

image_size = 224

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("model_folder_path")
    parser.add_argument("--top_k", dest="top_k", type=int, default=1)
    parser.add_argument("--category_names", dest="category_names", default="./label_map.json")
    args = parser.parse_args()

    print("args: ", args)
    

    image_path = args.image_path
    model_folder_path = args.model_folder_path
    top_k = args.top_k
    category_names = args.category_names
    model_folder_path = args.model_folder_path
    
    # Load class names
    with open(category_names, 'r') as f:
        class_names = json.load(f)

    # Load model and predict
    model = load_model(model_folder_path)
    (probs, classes) = predict(image_path, model, top_k)

    print("probs", probs)
    print("classes", classes)
    print("class_names", [class_names[str(i+1)] for i in classes])   
# Load model
def load_model(path):

    # Unknown layer: KerasLayer
    # See my note on Jupyter Notebook
    
    model = tf.keras.experimental.load_from_saved_model(path, custom_objects={'KerasLayer':hub.KerasLayer})
 
    #Get input shape from model.get_config()
    model.build((None, image_size,image_size, 3))
    return model


def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image,axis=0)
    ps = model.predict(processed_image)
    probs = ps[0]
    
    # Get k indexes of sorted probs
    top_indexes = np.argsort(probs)[-top_k:]
    
    # Reverse array
    top_indexes = top_indexes[::-1]
    
    probs.sort()
    # Reverse array
    probs = probs[-top_k:][::-1]
    return (probs,top_indexes)



if __name__ == "__main__":
    main()
