import os
import pandas as pd
import json
import tensorflow as tf

from object_detection.utils import dataset_util

path_to_json = 'labelme/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
path_to_jpg = 'labelme/'
jpg_files = [pos_jpg for pos_jpg in os.listdir(path_to_jpg) if pos_jpg.endswith('.jpg')]
fjpeg=(list(reversed(jpg_files)))
n=0
csv_list = []
labels=[]
for j in json_files:
    data_file=open('labelme/{}'.format(j))   
    data = json.load(data_file)
    height = data['imageHeight']
    width = data['imageWidth']
    filename = 'example_cat.jpg'
    image_format = b'jpg'

    xmins = [322.0 / 1200.0]
    xmaxs = [1062.0 / 1200.0]
    ymins = [174.0 / 1032.0]
    ymaxs = [761.0 / 1032.0]
    classes_text = ['Cat']
    classes = [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  