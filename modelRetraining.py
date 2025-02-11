# %%
import sys
sys.path.append("/home/gibrown/Google/models")
import os

from official import core
from official.core import exp_factory
from official import vision
from official.core import train_utils
from official.vision.modeling import factory
from official.vision.serving import export_saved_model_lib

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
logging.disable(logging.WARNING)
import wandb
wandb.login()

import argparse

# %%
import tensorflow_models as tfm
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
sys.path.append('models/official/projects/waste_identification_ml/model_inference/')
import postprocessing
import preprocessing
import labels

def build_model(task_config):
    """Builds Mask R-CNN model."""

    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + task_config.model.input_size)

    l2_weight_decay = task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = factory.build_maskrcnn(
        input_specs=input_specs,
        model_config=task_config.model,
        l2_regularizer=l2_regularizer)

    if task_config.freeze_backbone:
      model.backbone.trainable = False

    # Builds the model through warm-up call.
    dummy_images = tf.keras.Input(task_config.model.input_size)
    dummy_image_shape = tf.keras.layers.Input([2])
    _ = model(dummy_images, image_shape=dummy_image_shape, training=False)

    return model

# %%
def loadAndBuild(source_dir, tf_records_dir='tf_records'):
  param_file = source_dir+'/params.yaml'
  
  with open(param_file, 'r') as file:
    override_params = yaml.full_load(file)
  exp_config = exp_factory.get_exp_config('maskrcnn_resnetfpn_coco')
  exp_config.override(override_params, is_strict=False)
  
  train_data_input_path = BASE_DIR + "data/" + tf_records_dir + '/*'
  valid_data_input_path = BASE_DIR + "data/" + tf_records_dir + '/*'
  # valid_data_input_path = BASE_DIR + "data/tf_record_groups/tf_records_test/*"
  # test_data_input_path  = BASE_DIR + 'zerowaste-f-final/splits_final_deblurred/train/data/*'
  exp_config.task.train_data.input_path = train_data_input_path
  exp_config.task.validation_data.input_path = valid_data_input_path
  exp_config.task.train_data.global_batch_size = 8
  exp_config.task.train_data.shuffle_buffer_size = 100
  exp_config.task.validation_data.global_batch_size = 8
  exp_config.task.validation_data.shuffle_buffer_size = 100
  exp_config.task.annotation_file = BASE_DIR + "/data/tf_record_groups/hard_99.0.json"
  exp_config.task.init_checkpoint = source_dir + "/checkpoint/model.ckpt"
  exp_config.runtime.distribution_strategy = "GPU"
  exp_config.task.freeze_backbone = True
  return exp_config

# %%

parser = argparse.ArgumentParser(description="Retrains from a pretrained backbone")
parser.add_argument('--tfdata', default='tf_records', help="Directory relative to BASE_DIR/data/ to retrieve tf records, default = tf_records")
parser.add_argument('--savedir', default='new_checkpoint', help="Directory relative to autolabeled/ to save checkpoints. Will be created if it does not already exist, default = new_checkpoint")
parser.add_argument('--name', default='default_run', help="Name for human readable outputs and saving, please don't include any spaces. default = default_run")
args = parser.parse_args()

exp_config = loadAndBuild(BASE_DIR + "autolabeled", args.tfdata)
if(args.tfdata != 'tf_records'):
  exp_config.task.annotation_file = BASE_DIR + "data/" + args.tfdata + ".json"


# %%
model = build_model(exp_config.task)


checkpoint_dir = BASE_DIR + "autolabeled"
ckpt = tf.train.Checkpoint(backbone=model.backbone)
status = ckpt.read(os.path.join(checkpoint_dir, "checkpoint", "model.ckpt"))
status.expect_partial().assert_existing_objects_matched()
model.summary()

# %%
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


# %%
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


# %%
model_dir = BASE_DIR + "autolabeled"

with distribution_strategy.scope():
  task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)


# %%
ds = task.build_inputs(exp_config.task.train_data)

opt_config = tfm.optimization.OptimizationConfig(exp_config.get('trainer').get('optimizer_config'))
opt_factory = tfm.optimization.OptimizerFactory(opt_config)
lr = opt_factory.build_learning_rate()
optimizer = opt_factory.build_optimizer(lr)

# %%
trainer = tfm.core.base_trainer.Trainer(
  config= exp_config,
  task= task,
  model=model,
  optimizer= optimizer,
  train= True,
  evaluate= True
)

# %%
def write_to_checkpoint(model, epoch, dir):
  checkpoint_path = os.path.join(BASE_DIR, f"autolabeled/{dir}/ckpt-{epoch:04d}")
  checkpoint_dir = os.path.dirname(checkpoint_path)
  os.makedirs(checkpoint_dir, exist_ok=True)

  ckpt = tf.train.Checkpoint(model=model)
  ckpt.write(checkpoint_path.format(epoch=epoch))

def visualize_train_data(trainer):
  r = trainer.train_dataset.take(40)
  for i, (a,b) in enumerate(r):
    image_np = a
    predicted_mask = b
    height = trainer.config.task.model.input_size[0]
    width = trainer.config.task.model.input_size[1]
    image_np_cp = tf.image.resize(image_np, (height, width), method=tf.image.ResizeMethod.AREA)
    image_np_cp *= (255/np.max(image_np_cp))
    image_np_cp = tf.cast(image_np_cp, tf.uint8)
    # image_np = preprocessing.normalize_image(image_np_cp)
    image_np = image_np_cp.numpy()[0].astype(dtype=np.uint8)
    if(isinstance(predicted_mask['gt_boxes'], tf.Tensor)):
      predicted_mask['detection_boxes'] = predicted_mask['gt_outer_boxes'].numpy()
      predicted_mask['detection_masks'] = predicted_mask['gt_masks'].numpy()

    predicted_mask = postprocessing.reframing_masks(
        predicted_mask, height, width
    )

    masks = np.zeros(image_np.shape, dtype=np.uint8)
    for mask in predicted_mask['detection_masks_reframed']:
        if(np.count_nonzero(mask) > 5):
          m = mask.astype(dtype=np.uint8)*255
          m = np.stack([m,m,m],axis=2)
          masks = np.maximum(m, masks)
    im = cv2.addWeighted(image_np, .5, masks, .5, 0.0)
    #Replace with your directory names
    cv2.imwrite(f"/home/gibrown/Google/temp/train_masks/{i}.png", im)

def visualize_eval_data(trainer, run, name):
    r = trainer.eval_dataset.take(10)
    #Replace with your directory names
    os.makedirs(f"/home/gibrown/Google/temp/eval_masks/{name}/{run}", exist_ok=True)
    for i, (a,b) in enumerate(r):
      # print(f"------------{run}.{i}-------------", flush=True)
      image_np = a
      predicted_mask = b
      height = trainer.config.task.model.input_size[0]
      width = trainer.config.task.model.input_size[1]
      image_np_cp = tf.image.resize(image_np, (height, width), method=tf.image.ResizeMethod.AREA)
      image_np_cp *= (255/np.max(image_np_cp))
      image_np_cp = tf.cast(image_np_cp, tf.uint8)
      # image_np = preprocessing.normalize_image(image_np_cp)
      image_np = image_np_cp.numpy()[0].astype(dtype=np.uint8)
      # print(predicted_mask['groundtruths']['classes'].numpy(), flush=True)
      if(isinstance(predicted_mask['groundtruths']['boxes'], tf.Tensor)):
        predicted_mask['detection_boxes'] = predicted_mask['groundtruths']['boxes'].numpy()
        predicted_mask['detection_masks'] = predicted_mask['groundtruths']['masks'].numpy()

      predicted_mask = postprocessing.reframing_masks(
          predicted_mask, height, width
      )
      masks = np.zeros(image_np.shape, dtype=np.uint8)
      for mask in predicted_mask['detection_masks_reframed']:
          if(np.count_nonzero(mask) > 5):
            m = mask.astype(dtype=np.uint8)*255
            m = np.stack([m,np.zeros(m.shape, dtype=np.uint8), np.zeros(m.shape, dtype=np.uint8)],axis=2)
            masks = np.maximum(m, masks)

      predicted_mask_2 = trainer.model(preprocessing.normalize_image(image_np_cp), image_shape=[height, width])
      # print(predicted_mask_2['detection_classes'].numpy(), flush=True)
      if(isinstance(predicted_mask_2['detection_boxes'], tf.Tensor)):
        predicted_mask_2['detection_boxes'] = predicted_mask_2['detection_boxes'].numpy()
        predicted_mask_2['detection_masks'] = predicted_mask_2['detection_masks'].numpy()
      predicted_mask_2 = postprocessing.reframing_masks(
          predicted_mask_2, height, width
      )

      masks_2 = np.zeros(image_np.shape, dtype=np.uint8)
      for mask in predicted_mask_2['detection_masks_reframed']:
          if(np.count_nonzero(mask) > 5):
            m = mask.astype(dtype=np.uint8)*255
            m = np.stack([np.zeros(m.shape, dtype=np.uint8), np.zeros(m.shape, dtype=np.uint8), m],axis=2)
            masks_2 = np.maximum(m, masks_2)
      im = cv2.addWeighted(image_np, .5, masks+masks_2, .5, 0.0)
      cv2.imwrite(f"/home/gibrown/Google/temp/eval_masks/{name}/{run}/{i}.png", im)

# %%
wandb.init(
        # Set the project where this run will be logged
        project="ML_scrap", 
        entity="wpirecycles",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"{args.name}", 
        # Track hyperparameters and run metadata
        config={
        "architecture": "Mask-RCNN",
        "dataset": "WPI",
        })

visualize_train_data(trainer)
for run in range(140):
  wandb_dict = {"run": run}
  if(0 == (run % 5)):
    visualize_eval_data(trainer, run, str(args.name))
    ret2 = trainer.evaluate(tf.convert_to_tensor(512))
    for k, v in ret2.items():
      if('validation_loss' == k):
        v = v.numpy()
      wandb_dict[k] = v
    write_to_checkpoint(trainer.model, run, args.savedir)
  ret = trainer.train(tf.convert_to_tensor(512))
  for k, v in ret.items():
    wandb_dict[k] = v.numpy()
  #Debug code
  # if(0 == (run % 5) and run != 0):
  #   export_saved_model_lib.export_inference_graph(
  #     input_type='image_tensor',
  #     batch_size=1,
  #     input_image_size=[480, 640],
  #     params=exp_config,
  #     checkpoint_path=tf.train.latest_checkpoint(model_dir),
  #     export_dir=os.path.join(BASE_DIR, "autolabeled", "saved_model"))
  wandb.log(wandb_dict)
wandb.finish()


