import os
import pandas as pd
import json
import tensorflow as tf

from object_detection.utils import dataset_util

filenameout = 'train.record'
writer = tf.compat.v1.python_io.TFRecordWriter(filenameout)
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
    filename = data['imagePath']
    filenamebyte = filename.encode('utf8')
    image_format = b'jpg'
    x = data['shapes'][0]['points'][0]
    y = data['shapes'][0]['points'][2]
    xmins = [x[0] / height]
    xmaxs = [y[0] / height]
    ymins = [x[1] / width]
    ymaxs = [y[1] / width]
    classes_data = data['shapes'][0]['label']
    classes_text = []
    classes_text.append(classes_data.encode('utf8'))
    classes = [1]
    in_file = open('labelme/{}'.format(j), "rb")
    dataencode = in_file.read()

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filenamebyte),
      'image/source_id': dataset_util.bytes_feature(filenamebyte),
      'image/encoded': dataset_util.bytes_feature(dataencode),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  
    serialized_features_dataset = tf_example.SerializeToString()
    writer.write(serialized_features_dataset)
writer.close()