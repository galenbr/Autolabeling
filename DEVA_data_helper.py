from config import *
import os
from os import path
from torch import save, load, arange
import json
from deva.inference.object_info import ObjectInfo
from PIL import Image
import numpy as np
from deva.inference.data.detection_video_reader import DetectionVideoReader
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.inference.frame_utils import FrameInfo
from deva.inference.object_utils import convert_json_dict_to_objects_info
from torch import from_numpy, device


class frameHandler(object):
  def __init__(self, path, name="SAM_") -> None:
    self.path = path
    self.name = name
    os.makedirs(self.path, exist_ok=True)

  def saveFrame(self, frame_number, mask, segmentation_info=None):
    os.makedirs(path.join(self.path, str(frame_number)), exist_ok=True)
    mask_path = path.join(self.path, str(frame_number), self.name+"mask.pt")
    save(mask, mask_path)
    if(segmentation_info is None):
      return
    seg_path = path.join(self.path, str(frame_number), self.name+"seg.json")
    seg_dict = []
    for segmentation in segmentation_info:
      seg_dict.append({"id": segmentation.id, "category_id": segmentation.vote_category_id(), "score": segmentation.vote_score()})
    #https://stackoverflow.com/questions/17043860/how-to-dump-a-dict-to-a-json-file
    with open(seg_path, 'w') as fp:
      json.dump(seg_dict, fp)

  def checkMask(self, frame_number):
    mask_path = path.join(self.path, str(frame_number), self.name+"mask.pt")
    return os.path.exists(mask_path)
  
  def checkSeg(self, frame_number):
    seg_path = path.join(self.path, str(frame_number), self.name+"seg.json")
    return os.path.exists(seg_path)

  def loadFrame(self, frame_number):
    mask_path = path.join(self.path, str(frame_number), self.name+"mask.pt")
    seg_path = path.join(self.path, str(frame_number), self.name+"seg.json")
    if not os.path.exists(mask_path):
      raise Exception(f"{mask_path} does not exist!")
    if not "SAM_" == self.name and not os.path.exists(seg_path):
      raise Exception(f"{mask_path} does not exist!")
    mask = load(mask_path)
    with open(seg_path, 'r') as fp:
      raw_seg = json.load(fp)
    output = [
            ObjectInfo(
                id=segment['id'],
                category_id=segment.get('category_id'),
                isthing= None,
                score=float(segment['score']))
            for segment in raw_seg
        ]
    return mask, output
  

class interHandler(object):
  def __init__(self, img_path:str, mask_path:str, label_path:str, size:int=480) -> None:
    self.img_path = img_path
    self.mask_path = mask_path
    self.label_path = label_path
    self.size = size
    self.loaders = {}
    self.offset = 0

  def _checkLoader(self, prefix):
    if prefix not in self.loaders.keys():
      root_dir = path.join(self.mask_path, prefix)
      vr = DetectionVideoReader(
                prefix,
                self.img_path,
                path.join(root_dir, prefix),
                to_save=None,
                size=self.size,
            )
      self.loaders[prefix] = vr 

  def loadSegment(self, prefix, frame_name):
    json_file = path.join(self.label_path, prefix, frame_name[:-4]+".json")
    try:
      with open(json_file, 'r') as f:
        all_json_info = json.load(f)
    except FileNotFoundError:
      return None
    return all_json_info
  
  def saveSegment(self, prefix, frame_name, segment):
    json_file = path.join(self.label_path, prefix, frame_name[:-4]+".json")
    os.makedirs(path.join(self.label_path, prefix), exist_ok=True)
    with open(json_file, 'w') as f:
      json.dump(segment, f)

  def checkFrame(self, prefix, frame_name):
    root_dir = path.join(self.mask_path, prefix)
    file_name = path.join(root_dir, frame_name[:-4] + '.png')
    return path.exists(file_name)
  
  def getSeqLen(self, prefix):
    self._checkLoader(prefix=prefix)
    return len(self.loaders[prefix].frames)

  def saveFrame(self, prefix, frame_name, mask, override=False):
    #If we're not in override and the file already exists, just leave it.
    if override and self.checkFrame(prefix=prefix, frame_name=frame_name):
      return
    root_dir = path.join(self.mask_path, prefix)
    file_name = path.join(root_dir, frame_name[:-4] + '.png')
    os.makedirs(root_dir, exist_ok=True)
    out_img = Image.fromarray(mask.cpu().numpy().astype(np.uint8))
    out_img.save(file_name)

  def loadMask(self, prefix, frame_name):
    self._checkLoader(prefix=prefix)
    try:
      index = self.loaders[prefix].frames.index(frame_name[:-4]+'.png')
    except ValueError:
      index = self.loaders[prefix].frames.index(frame_name[:-4]+'.jpg')
    data = [self.loaders[prefix][index]]
    data[0]['mask'][:20,:] = 0
    return from_numpy(data[0]['mask']).to(device('cuda:0'))

  def loadFrame(self, prefix, frame_name):
    self._checkLoader(prefix=prefix)
    try:
      index = self.loaders[prefix].frames.index(frame_name[:-4]+'.png')
    except ValueError:
      index = self.loaders[prefix].frames.index(frame_name[:-4]+'.jpg')
    data = [self.loaders[prefix][index]]
    return data[0]['raw']
  
  def produceFrameInfo(self, prefix, frame_name, ti, frame_number=-1):
    if(-1==frame_number):
      frame_number = ti
    image_np = self.loadFrame(prefix=prefix, frame_name=frame_name)
    image_np[:20,:,:] = 0
    h, w = image_np.shape[:2]
    new_min_side = self.size
    image = get_input_frame_for_deva(image_np, new_min_side)

    frame_info = FrameInfo(image, None, None, ti, {
        'frame': [frame_name],
        'shape': [h, w],
        'id': [str(frame_number)]
    })
    if(self.checkFrame(prefix="", frame_name=frame_name)):
      frame_info.mask = self.loadMask(prefix="", frame_name=frame_name)
      segments_info = self.loadSegment(prefix="", frame_name=frame_name)
      frame_info.segments_info = convert_json_dict_to_objects_info(frame_info.mask,
                          segments_info,
                          dataset="coco")
      frame_info.image_np = image_np
      frame_info.frame_no = frame_number
    else:
      raise KeyError(f"{frame_name} not found!")
    return frame_info

class startInterHandler(interHandler):
  def _checkLoader(self, prefix):
    if prefix not in self.loaders.keys():
      root_dir = path.join(self.mask_path, prefix)
      vr = DetectionVideoReader(
                prefix,
                self.img_path,
                self.img_path,
                to_save=None,
                size=self.size,
            )
      self.loaders[prefix] = vr 
      
class offsetHandler(interHandler):
  def __init__(self, img_path: str, mask_path: str, label_path: str, size: int = 480, offset: int = 1) -> None:
    super().__init__(img_path, mask_path, label_path, size)
    self.offset = offset
    self.files = {}

  def __init__(self, base_handler: interHandler, offset: int = 1):
    super().__init__(base_handler.img_path, base_handler.mask_path, base_handler.label_path, base_handler.size)
    self.offset = offset
    self.files = {}

  def _checkFiles(self, prefix):
    if prefix not in self.files.keys():
      filenames = self.loaders[prefix].frames
      trimmed = [x[:-4] for x in filenames]
      self.files[prefix] = trimmed
  
  def produceFrameInfo(self, prefix, frame_name, ti, frame_no):
    self._checkLoader(prefix=prefix)
    self._checkFiles(prefix=prefix)
    new_frame_no = frame_no + self.offset
    target_ti = ti + self.offset
    new_target_index = self.files[prefix].index(frame_name[:-4]) - self.offset
    print(f"Trying to use {new_target_index}, {target_ti}, {new_frame_no}")
    if(new_target_index < 0 or new_target_index > len(self.files[prefix])-1):
      return None
    ret = super().produceFrameInfo(prefix, self.files[prefix][new_target_index]+".tmp", target_ti, new_frame_no)
    return ret


  def primeOffset(self, prefix, offset, target_frame_name):
    self.offset = offset
    self.target = target_frame_name