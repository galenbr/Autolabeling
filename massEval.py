

import sys
sys.path.append("/home/rbe07/Documents/Google/models/")
import os
import eta.core.utils as etau

from official import core
from official.core import exp_factory
from official import vision
from official.core import train_utils
from official.vision.modeling import factory

import yaml
import pprint
from urllib.request import urlopen
import tensorflow as tf, tf_keras
#tf.compat.v1.disable_eager_execution()
from PIL import Image
from six import BytesIO
import numpy as np
from typing import Sequence
import matplotlib.pyplot as plt
import cv2

from config import *
pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation

import logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logging.disable(logging.WARNING)
# import wandb
# wandb.login()

import tensorflow_models as tfm
# from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder


#https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
# import importlib.util
# spec = importlib.util.spec_from_file_location("color_and_property_extractor", "/home/rbe07/Documents/Google/models/official/projects/waste_identification_ml/model_inference/color_and_property_extractor.py")
# color_and_property_extractor = importlib.util.module_from_spec(spec)
# sys.modules["color_and_property_extractor"] = color_and_property_extractor
# spec.loader.exec_module(color_and_property_extractor)
# spec = importlib.util.spec_from_file_location("postprocessing", "/home/rbe07/Documents/Google/models/official/projects/waste_identification_ml/model_inference/postprocessing.py")
# postprocessing = importlib.util.module_from_spec(spec)
# sys.modules["postprocessing"] = postprocessing
# spec.loader.exec_module(postprocessing)

sys.path.append('models/official/projects/waste_identification_ml/model_inference/')
import color_and_property_extractor
import labels
import postprocessing
import preprocessing

import fiftyone as fo
import pandas as pd
# HEIGHT = 480
HEIGHT = 512
# HEIGHT = 160
# WIDTH = 160
WIDTH = 640



def convert_bb(bb):
    """Converts CN bounding boxes to VS51 format.
    CN output is [ymin, xmin, ymax, xmax], V51 wants [xmin, ymin, width%, height%].

    Args:
      bb: Bounding Boxes in CN format.
    
    Returns: 
      output: Bounding Boxes in V51 format.
    """
    output = [bb[1], bb[0], bb[3]-bb[1], bb[2]-bb[0]]
    return output



def build_model(task_config):
    """Builds Mask R-CNN model."""

    input_specs = tf_keras.layers.InputSpec(
        shape=[None] + task_config.model.input_size)

    l2_weight_decay = task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf_keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_maskrcnn(
        input_specs=input_specs,
        model_config=task_config.model,
        l2_regularizer=l2_regularizer,)
    
    # ckpt = tf.train.Checkpoint(model.backbone)
    # status = ckpt.read(MODEL_SOURCE)
    # status.expect_partial().assert_existing_objects_matched()

    # model = factory.build_maskrcnn(
    #     input_specs=input_specs,
    #     model_config=task_config.model,
    #     l2_regularizer=l2_regularizer,
    #     backbone=model.backbone)

    if task_config.freeze_backbone:
      model.backbone.trainable = False

    # Builds the model through warm-up call.
    dummy_images = tf_keras.Input(task_config.model.input_size)
    dummy_image_shape = tf_keras.layers.Input([2])
    _ = model(dummy_images, image_shape=dummy_image_shape, training=False)

    return model


def initialize(task_config, model):
    """Loads pretrained checkpoint."""

    if not task_config.init_checkpoint:
      return

    ckpt_dir_or_file = task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    # Restoring checkpoint.
    if task_config.init_checkpoint_modules == 'all':
      ckpt = tf.train.Checkpoint(model=model)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
    else:
      ckpt_items = {}
      if 'backbone' in task_config.init_checkpoint_modules:
        ckpt_items.update(backbone=model.backbone)
      if 'decoder' in task_config.init_checkpoint_modules:
        ckpt_items.update(decoder=model.decoder)

      ckpt = tf.train.Checkpoint(**ckpt_items)
      status = ckpt.read(ckpt_dir_or_file)
      status.expect_partial().assert_existing_objects_matched()
      print(f"Finished loading checkpoint {ckpt_dir_or_file}")
      
def loadAndBuild(source_dir, model_source):
  param_file = source_dir+'/params.yaml'
  
  with open(param_file, 'r') as file:
    override_params = yaml.full_load(file)
  exp_config = exp_factory.get_exp_config('maskrcnn_resnetfpn_coco')
  exp_config.override(override_params, is_strict=False)
  
  train_data_input_path = BASE_DIR + 'data/tf_records/*'
  valid_data_input_path = BASE_DIR + 'data/tf_records_test/*'
  test_data_input_path  = BASE_DIR + 'data/tf_records_test/*'
  exp_config.task.train_data.input_path = train_data_input_path
  exp_config.task.validation_data.input_path = valid_data_input_path
  exp_config.task.train_data.global_batch_size = 1
  exp_config.task.train_data.shuffle_buffer_size = 100
  exp_config.task.validation_data.global_batch_size = 1
  exp_config.task.validation_data.shuffle_buffer_size = 100
  exp_config.task.annotation_file = BASE_DIR + "data/5_5/Labels/CardboardOcclusions_both_2_post_labels_DEVA.json"
  exp_config.task.init_checkpoint = model_source
  # exp_config.task.init_checkpoint = None
  # exp_config.task.init_checkpoint = source_dir + "/s_checkpiont_a/ckpt-0309"
  exp_config.runtime.distribution_strategy = "GPU"
  # exp_config.task.freeze_backbone = True
  exp_config.task.model.input_size = [HEIGHT, WIDTH, 3]
  # exp_config.task.model.detection_head.fc_dims = WIDTH
  return exp_config


def process_single_image(model, image, directory_path, all_samples):
  image_path = os.path.join(directory_path, image)
  image_np = cv2.imread(image_path)
  image_np_cp = tf.image.resize(image_np, (HEIGHT, WIDTH), method=tf.image.ResizeMethod.AREA)
  image_np_cp = tf.cast(image_np_cp, tf.uint8)
  image_np = preprocessing.normalize_image(image_np_cp)
  # image_np = image_np_cp
  image_np = tf.expand_dims(image_np, axis=0)
#   image_np = cv2.resize(image_np, (WIDTH, HEIGHT))
  # mask = tf.image.resize(image_np, size=[HEIGHT, WIDTH])
  i_s = [HEIGHT, WIDTH]
  predicted_mask = model(image_np, image_shape=i_s)
#   predicted_mask = model(image_np)

  if(isinstance(predicted_mask['detection_boxes'], tf.Tensor)):
      predicted_mask['detection_boxes'] = predicted_mask['detection_boxes'].numpy()
      predicted_mask['detection_masks'] = predicted_mask['detection_masks'].numpy()

  # for bb in predicted_mask['detection_boxes'][0]:
    #  print(bb)
  predicted_mask = postprocessing.reframing_masks(
      predicted_mask, HEIGHT, WIDTH
  )

  ind = 0
  # for mask in predicted_mask['detection_masks_reframed']:
  #     m = mask.astype(dtype=np.uint8)*255
  #     m = np.stack([m,m,m],axis=2)
  #   #   print(image_np.shape)
  #     im = cv2.addWeighted(image_np.numpy()[0].astype(dtype=np.uint8), .5, m, .5, 0.0)
  #     cv2.imwrite("/home/rbe07/Downloads/temp/"+str(ind)+".png", im)
  #     ind += 1

  transformed_boxes = []
  for bb in predicted_mask['detection_boxes'][0]:
      YMIN = int(bb[0]*HEIGHT)
      XMIN = int(bb[1]*WIDTH)
      YMAX = int(bb[2]*HEIGHT)
      XMAX = int(bb[3]*WIDTH)
      transformed_boxes.append([YMIN, XMIN, YMAX, XMAX])
      # print(bb)
      # print([YMIN, XMIN, YMAX, XMAX])

  # print(predicted_mask['detection_boxes'][0])
  # Filtering duplicate bounding boxes.
  filtered_boxes, index_to_delete = (
      postprocessing.filter_bounding_boxes(transformed_boxes))

  final_result = {}
  final_result['num_detections'] = predicted_mask['num_detections'][0] - len(index_to_delete)
  final_result['detection_classes'] = np.delete(
      predicted_mask['detection_classes'].numpy(), index_to_delete)
  final_result['detection_scores'] = np.delete(
      predicted_mask['detection_scores'].numpy(), index_to_delete, axis=1)
  final_result['detection_boxes'] = np.delete(
      predicted_mask['detection_boxes'], index_to_delete, axis=1)
  # final_result['detection_classes_names'] = np.delete(
  #     predicted_mask['detection_classes_names'].numpy(), index_to_delete)
  final_result['detection_masks_reframed'] = np.delete(
      predicted_mask['detection_masks_reframed'], index_to_delete, axis=0)
  detections = []
  if(final_result['num_detections'] > 0):
      # Calculate properties of each object for object tracking purpose.
      dfs, cropped_masks = (
          color_and_property_extractor.extract_properties_and_object_masks(
              final_result, HEIGHT, WIDTH, image_np_cp))
      features = pd.concat(dfs, ignore_index=True)
      features['image_name'] = "Temp_name"
      features.rename(columns={
          'centroid-0':'y',
          'centroid-1':'x',
          'bbox-0':'bbox_0',
          'bbox-1':'bbox_1',
          'bbox-2':'bbox_2',
          'bbox-3':'bbox_3'
      }, inplace=True)

      
      dominant_colors = [*map(color_and_property_extractor.find_dominant_color, cropped_masks)]
      color_names = [*map(color_and_property_extractor.get_color_name, dominant_colors)]
      for i in range(final_result["num_detections"].numpy()):
          # l = final_result['detection_classes_names'][i]
        #   l = str(final_result['detection_classes'][i])
          bb = convert_bb(final_result['detection_boxes'][0][i])
          m = np.array(cropped_masks[i], dtype=bool)
          c = final_result['detection_scores'][0][i]
          if(m.sum() > 5):
            detections.append(
                fo.Detection(
                    label="cardboard",
                    bounding_box=bb,
                    mask=m[:,:,0],
                    confidence=c
                )
            )
          else:
            print(f"Error in {image}")
  current_sample = fo.Sample(filepath = image_path)
  current_sample["predictions"] = fo.Detections(detections=detections)
  all_samples.append(current_sample)

def process_images(model, images_list, directory_path):
    all_samples = []
    with etau.ProgressBar(len(images_list)) as pb:
        for image in images_list:
            process_single_image(model, image=image, directory_path=directory_path, all_samples=all_samples)
            pb.update()
    return all_samples

def mainLoop(ds_name, image_source_dir):
  MODEL_SOURCE = f"/home/rbe07/Documents/Google/remote_model/{ds_name}/ckpt-0050"
  MODEL_DIR = f"{BASE_DIR}remote_model"
  exp_config = loadAndBuild(MODEL_DIR, MODEL_SOURCE)

  model = build_model(exp_config.task)
  logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

  if 'GPU' in ''.join(logical_device_names):
    print('This may be broken in Colab!')
    device = 'GPU'
  elif 'TPU' in ''.join(logical_device_names):
    print('This may be broken in Colab')
    device = 'TPU'
  else:
    print('Running on CPU is slow, so only train for a few steps.')
    device = 'CPU'

  # Setting up the Strategy
  if exp_config.runtime.mixed_precision_dtype == tf.float16:
      tf.keras.mixed_precision.set_global_policy('mixed_float16')

  if 'GPU' in ''.join(logical_device_names):
    distribution_strategy = tf.distribute.MirroredStrategy()
  elif 'TPU' in ''.join(logical_device_names):
    tf.tpu.experimental.initialize_tpu_system()
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='/device:TPU_SYSTEM:0')
    distribution_strategy = tf.distribute.experimental.TPUStrategy(tpu)
  else:
    print('Warning: this will be really slow.')
    distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

  print("Done")

  with distribution_strategy.scope():
    task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=MODEL_DIR)

  ds = task.build_inputs(exp_config.task.train_data)

  opt_config = tfm.optimization.OptimizationConfig(exp_config.get('trainer').get('optimizer_config'))
  opt_factory = tfm.optimization.OptimizerFactory(opt_config)
  lr = opt_factory.build_learning_rate()
  optimizer = opt_factory.build_optimizer(lr)

  trainer = tfm.core.base_trainer.Trainer(
    config= exp_config,
    task= task,
    model=model,
    optimizer= optimizer,
    train= True,
    evaluate= True
  )
  trainer.initialize()
  current_dataset = fo.Dataset(ds_name, overwrite=True)
  images_list = os.listdir(image_source_dir)
  all_samples = process_images(model, images_list=images_list, directory_path=image_source_dir)
  current_dataset.add_samples(all_samples)

  # Export the dataset
  current_dataset.export(
      labels_path=BASE_DIR + "datasets/" + ds_name + ".json",
      dataset_type=fo.types.COCODetectionDataset,
      label_field='predictions',
  )

if __name__ == '__main__':
  # image_source_dir = "/home/rbe07/Documents/Google/data/sequences/hand_labeled_corrected"
  image_source_dir = "/home/rbe07/Documents/Google/data/sequences/A_hand"
  data_sources = ["0", "50", "90", "99"]
  # data_sources = ["CO_corrected_ckpt0", "CO_orig_ckpt0", "CO_ckpt0"]
  for data_source in data_sources:
     mainLoop(data_source, image_source_dir)