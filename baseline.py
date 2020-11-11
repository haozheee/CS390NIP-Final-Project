import tensorflow as tf
import collections
import random
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle


def getRawData():
    annotation_folder = '/annotations/'
    annotation_file = ''
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath('.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)

    # Download image files
    image_folder = '/train2014/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('.') + image_folder

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)
    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)
    return image_paths, image_path_to_caption

def preprocessData(raw):
    image_paths, image_path_to_caption = raw

    # before pre-processing, each image is corresponding to multiple caption.
    # we will duplicate the images so that we have (image, caption) pairs

    train_captions = []
    img_name_vector = []
    for image_path in image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))
    encode_train = sorted(set(img_name_vector))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    # pretrained InceptionV3 to extract features from images
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # cache the features of image extracted by InceptionV3 to the disk
    # because the memory in RAM is not sufficient to store these features for all images
    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    # img_name_vector is a list of image file paths
    # train_captions is a list of corresponding captions
    # we need to split the training and testing set from this
    return img_name_vector, train_captions

def trainModel(train_x, train_y):
    # TODO: complete the model
    pass


def runModel(test_x, test_y):
    # TODO: complete the model
    pass


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def main():
    print("Image Captioning Main")
    raw = getRawData()
    data_x, data_y = preprocessData(raw)
    model = trainModel(data_x, data_y)
    '''preds = runModel(data_, model)
    evalResults(data[1], preds)

    # classification on IRIS
    print("Classification on IRIS")
    raw = getRawDataIris()
    data = preprocessDataIris(raw)
    model = trainModelIris(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)'''


if __name__ == '__main__':
    main()
