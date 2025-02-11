import os
from os import path
from argparse import ArgumentParser
from config import *
import sys
from copy import copy
from deva.inference.postprocess_unsup_davis17 import limit_max_id

sys.path.append(os.path.join(BASE_DIR, "Tracking-Anything-with-DEVA"))

import torch
import numpy as np
from typing import Dict, List
import cv2

from deva.model.network import DEVA
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.result_utils import ResultSaver
from deva.inference.eval_args import add_common_eval_args, get_model_and_config
from deva.inference.demo_utils import flush_buffer
from deva.ext.ext_eval_args import add_ext_eval_args, add_text_default_args
from deva.ext.grounding_dino import get_grounding_dino_model
from deva.ext.with_text_processor import process_frame_with_text as process_frame
from deva.inference.object_info import ObjectInfo
from deva.inference.inference_core import DEVAInferenceCore
from deva.inference.frame_utils import FrameInfo
from deva.inference.result_utils import ResultSaver
from deva.inference.demo_utils import get_input_frame_for_deva
from deva.utils.palette import davis_palette
from deva.ext.automatic_processor import make_segmentation, estimate_forward_mask
from deva.ext.automatic_sam import get_sam_model

from videoHandling import revVid, combineVideos

from time import sleep

from tqdm import tqdm
import json

torch.autograd.set_grad_enabled(False) 

from DEVA_data_helper import interHandler, offsetHandler

from sklearn.cluster import DBSCAN
from itertools import combinations
import ipdb

all_masks = None
sam_model = None
save_decisions = {}

def _get_centroid(mask, id):
    m = mask.cpu().data.numpy().astype(np.uint8)
    m[m!=id] = 0
    square_mask = np.nonzero(m)
    area = 0
    if(len(square_mask[0]) > 0):
        mins = np.min(square_mask, axis=1)
        maxs = np.max(square_mask, axis=1)
        total = maxs - mins
        area = (total[0] * total[1]) / m.size
    # area = np.count_nonzero(m) / m.size
    if(0 >= np.max(m)):
        return [-1,-1], area
    M = cv2.moments(m)
    x = M["m10"]/M["m00"]/mask.shape[1]
    y = M["m01"]/M["m00"]/mask.shape[0]
    return [x, y], area

def findMotionTrend(obj, length):
    centroids = obj.get_all_centroids()
    length = min(length, len(centroids))
    if(0 == length):
        return np.array([0,0])
    ret = np.diff(np.array(centroids), axis=0)[-length:]
    return np.mean(ret, axis=0)

#Used for weak supervision- a human classifies every new detection
def okSave(mask, tmp_id, id, image):
    if id in save_decisions.keys():
        return save_decisions[id]
    m = mask.cpu().data.numpy().astype(np.uint8)
    m[m!=tmp_id] = 0
    m *= int(255/max(1, np.max(m)))
    im = np.stack([m,m,m],axis=2)
    if(image is not None):
        im = cv2.addWeighted(image, .5, im, .5, 0.0)
    cv2.imshow("Mask+image", im)
    r = -1
    while(ord('q') != r and ord('e') != r):
        print(f"Waiting for human input on {id}")
        r = cv2.waitKey()
        if(ord('q') == r):
            print(f"Rejecting {id}")
            save_decisions[id] = False
            return False
        if(ord('e') == r):
            print(f"Saving {id}")
            save_decisions[id] = True
            return True
        print(r)

def dummyOK(mask, tmp_id, id):
    return True

def updateCentroids(prob, deva, image=None):
    global all_masks
    mask = torch.argmax(prob, dim=0)
    purge = []
    purge_tmp = []
    keep = []
    path_mask = torch.zeros((mask.shape[0], mask.shape[1]), dtype = torch.uint8)
    obj_dict = deva.object_manager.obj_to_tmp_id.items()
    rejected = False
    for obj, tmp_id in obj_dict:
        [x, y], a = _get_centroid(mask, tmp_id)
        obj.set_centroid(x, y)
        if(
           .2 <= obj.max_disp):
            #https://stackoverflow.com/questions/62372762/delete-an-element-from-torch-tensor
            purge.append(obj.id)
            purge_tmp.append(tmp_id)
            mask[tmp_id==mask] = 0
            # print(f"Purging {tmp_id}, {np.max(obj.recent_disp)}, {1e-4 >= obj.recent_disp[0] and 1e-4 >= obj.recent_disp[1]}")
        else:
            keep.append(tmp_id-1)
            if(a >= .8):
                if(obj.save):
                    print(f"Rejecting {tmp_id} for large area")
                    rejected = True
                obj.save = False
            #Uncomment to enter weak supervision mode
            # obj.save = okSave(mask, tmp_id=tmp_id, id=obj.id, image=image)
            if(obj.save):
                path = torch.where(mask == tmp_id, 1, 0)
                path_mask = torch.maximum(path_mask.cpu(), path.cpu())

    if(rejected):
        for obj, tmp_id in obj_dict:
            print(f"{tmp_id}: {obj.save}")
    
    prob = purge_DEVA(prob=prob, purge=purge, purge_tmp=purge_tmp, keep=keep, deva=deva)
    
    #Delete the oldest historical mask and add the new one
    all_masks = torch.roll(all_masks, shifts=1, dims=0)
    all_masks[-1, :, :] = path_mask
    #OR the masks together
    c_path_mask = torch.amax(all_masks, dim=0)
    # https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate
    kernel = torch.ones((1,1,15,15), dtype=torch.uint8)
    # cv2.imshow("before", c_path_mask.numpy().astype(dtype=np.uint8)*40)
    c_path_mask = 1-torch.clamp(torch.nn.functional.conv2d(1-c_path_mask.unsqueeze(0).unsqueeze(0), kernel, padding=(7,7)), 0, 1).squeeze(0).squeeze(0)
    # cv2.imshow("after", c_path_mask.numpy().astype(dtype=np.uint8)*40)
    #Remove all points that are not on the paths
    c_path_mask = torch.mul(mask.cpu(), c_path_mask)
    
    for obj, tmp_id in obj_dict:
        if obj.save or 2 <= len(obj.get_all_centroids()):
            continue
        #If the mask overlaps one of the paths, mark it to be saved.
        if tmp_id in c_path_mask:
            if(obj.get_centroid()[0] >= .1):
                obj.save = True
                # obj.save = okSave(mask, tmp_id=tmp_id, id=obj.id, image=image)
            else:
                print(f"Centroid {obj.get_centroid()} rejected")
            # print(f"Object {tmp_id}, {obj.id} saved due to mask proximity")
            # cv2.imshow("Masks", c_path_mask.numpy().astype(dtype=np.uint8)*10)
            # cv2.waitKey()
    return prob

def purge_DEVA(prob, purge, purge_tmp, keep, deva):
    if(0 < len(purge)):
        #Debug code
        # cpu_mask = torch.argmax(prob, dim=0).cpu().data.numpy().astype(dtype=np.uint8)
        # # cpu_mask = np.expand_dims(cpu_mask, 2)
        # # cpu_mask = cv2.applyColorMap(cpu_mask, cv2.COLORMAP_JET)
        # cv2.imshow("final", cpu_mask)
        # cv2.waitKey()

        #We need to remove the purged masks at the end, in reverse order, or the tmp ids get all messed up.
        purge_tmp.sort(reverse=True)
        for tmp_id in purge_tmp:
            prob = torch.cat((prob[:tmp_id, :, :], prob[tmp_id+1:, :, :]), dim=0)
        deva.object_manager.delete_object(purge)
        retain = [obj.id for obj in deva.object_manager.obj_to_tmp_id.keys()]
        deva.memory.purge_except(retain)
        deva.last_mask = deva.last_mask.squeeze(0)[keep].unsqueeze(0)
    return prob

@torch.inference_mode()
def process_frame_with_text(deva: DEVAInferenceCore,
                            frame_name: str,
                            result_saver: ResultSaver,
                            ti: int,
                            source_handler: interHandler = None,
                            dir: int = 1,
                            mask_handlers: List[interHandler] = [],
                            frame_no: int = -1) -> None:
    # image_np, if given, should be in RGB
    cfg = deva.config
    # raw_prompt = cfg['prompt']
    prompts = None
    if('prompt' in cfg.keys() and cfg['prompt'] is not None):
      prompts = cfg['prompt'].split('.')
    if('prompt' in cfg.keys() and cfg['prompt'] is not None):
      prompts = cfg['prompt'].split('.')

    new_min_side = cfg['size']
    need_resize = new_min_side > 0

    frame_info = source_handler.produceFrameInfo("", frame_name=frame_name, ti=ti)
    h, w = frame_info.image_np.shape[:2]
    frame_info.frame_no = frame_no
    global all_masks
    if(all_masks is None):
        all_masks = torch.zeros((10, h, w), dtype=torch.uint8)

    if cfg['temporal_setting'] == 'semionline':
        if ti + cfg['num_voting_frames'] > deva.next_voting_frame:
            deva.add_to_temporary_buffer(frame_info)

            if ti == deva.next_voting_frame:
                # process this clip
                this_image = deva.frame_buffer[0].image
                this_frame_name = deva.frame_buffer[0].name
                this_image_np = deva.frame_buffer[0].image_np
                this_no = deva.frame_buffer[0].frame_no
                added_frames = 0
                for mask_h in mask_handlers:
                    new_frame = mask_h.produceFrameInfo("", this_frame_name, deva.frame_buffer[0].ti, this_no)
                    if(new_frame is not None):
                        deva.add_to_temporary_buffer(new_frame)
                        added_frames += 1
                _, mask, new_segments_info = deva.vote_in_temporary_buffer(
                    keyframe_selection='first')
                prob = deva.incorporate_detection(this_image, mask, new_segments_info)
                prob = updateCentroids(prob=prob, deva=deva, image=None)
                deva.next_voting_frame += cfg['detection_every']
                result_saver.save_mask(prob,
                                       this_frame_name,
                                       need_resize=need_resize,
                                       shape=(h, w),
                                       image_np=this_image_np,
                                       prompts=prompts,
                                       frame_no=this_no)
                #Remove the added frames
                deva.frame_buffer = deva.frame_buffer[:(len(deva.frame_buffer)-added_frames)]

                
                for old_frame in deva.frame_buffer[1:]:
                    this_image = old_frame.image
                    this_frame_name = old_frame.name
                    this_image_np = old_frame.image_np
                    prob = deva.step(this_image, None, None)
                    prob = updateCentroids(prob=prob, deva=deva, image=None)
                    result_saver.save_mask(prob,
                                           this_frame_name,
                                           need_resize,
                                           shape=(h, w),
                                           image_np=this_image_np,
                                           prompts=prompts,
                                           frame_no=old_frame.frame_no)

                #Remove detection_every frames.
                for f in deva.frame_buffer[:cfg['detection_every']]:
                    deva.image_feature_store.delete(f.ti)
                deva.frame_buffer = deva.frame_buffer[cfg['detection_every']:]
        else:
            # standard propagation
            prob = deva.step(frame_info.image, None, None)
            prob = updateCentroids(prob=prob, deva=deva, image=None)
            result_saver.save_mask(prob,
                                   frame_name,
                                   need_resize=need_resize,
                                   shape=(h, w),
                                   image_np=frame_info.image_np,
                                   prompts=prompts,
                                   frame_no=frame_info.frame_no)

    return None, None

def singleRun(cfg, source_handler: interHandler, sequence, frame_names, prefix="", suffix="", reversed=False, mask_handlers: List[interHandler] = [], input_deva=None):
    cfg['DINO_THRESHOLD'] = 0.25

    # Load our checkpoint
    if(input_deva is None):
        deva_model = DEVA(cfg).cuda().eval()
        if args.model is not None:
            model_weights = torch.load(args.model)
            deva_model.load_weights(model_weights)
        else:
            print('No model loaded.')
        deva = DEVAInferenceCore(deva_model, config=cfg)
    
        deva.enabled_long_id()
    else:
        deva = input_deva

    sequence_labels = {}

    f = open(os.path.join(BASE_DIR, DATA_DIR, "sequences", "Labels", sequence+"_labels.json"))
    sequence_labels = json.load(f)
    f.close()
    sequence_labels['annotations'] = []

    
    OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, DATA_DIR, prefix, "DEVA_output"+ suffix, sequence) 
    sink_handler = interHandler(source_handler.img_path, OUTPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, INT_SIZE)

    deva.next_voting_frame = cfg['num_voting_frames'] - 1 - len(mask_handlers)

    # obtain temporary directory
    just_save_final = not ('_fwd'==suffix or '_rev'==suffix)
    just_save_final = True
    result_saver = ResultSaver(os.path.join(BASE_DIR, DATA_DIR, prefix, "DEVA_output"), OUTPUT_VIDEO_PATH, dataset='coco', object_manager=deva.object_manager, palette=davis_palette, just_final=just_save_final)
    print(f"Only saving final value = {just_save_final}")

    ti = 0

    # Loop vars
    frame_no = 0
    annotations = []
    length = source_handler.getSeqLen("")
    frame_update = 1
    if(reversed):
        frame_no = length -1
        frame_update = -1

    frame_no += (BEGIN_AT*frame_update)
    length -= BEGIN_AT

    for i in tqdm(range(length)):
      mask, segments_info, = process_frame_with_text(deva,
                                  frame_names[frame_no][:-4]+'.png',
                                  result_saver,
                                  ti,
                                  source_handler=source_handler,
                                  dir=frame_update,
                                  mask_handlers=mask_handlers,
                                  frame_no = frame_no)
      annotations.append((mask, segments_info))
      ti += 1
      frame_no += frame_update
      
    flush_buffer(deva, result_saver)
    deva.clear_buffer()
    result_saver.end()
    global all_masks
    all_masks = None

    sequence_labels["annotations"] = result_saver.all_annotations
    cats = sequence_labels["categories"]
    ins = {"id": 7, "name": "merge", "supercategory": ""}
    if ins not in cats:
        cats.append(ins)
        sequence_labels["categories"] = cats
    by_frame = {}
    sleep(5)
    for annotation in sequence_labels["annotations"]:
        frame_name = frame_names[annotation["image_id"]][:-4]+'.png'
        if(frame_name not in by_frame.keys()):
            by_frame[frame_name] = []
        
        print(frame_name, end=":")
        print(annotation["image_id"], end=",")
        print(sequence_labels["images"][annotation["image_id"]]["id"])
        annotation["image_id"] = sequence_labels["images"][annotation["image_id"]]["id"]
        by_frame[frame_name].append(annotation)
    for frame_name in by_frame.keys():
        sink_handler.saveSegment(prefix="", frame_name=frame_name, segment=by_frame[frame_name])

    
    OUTPUT_LABELS_DIR = os.path.join(BASE_DIR, DATA_DIR, prefix, "Labels")
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    with open(path.join(OUTPUT_LABELS_DIR, sequence+suffix+"_labels_DEVA.json"), 'w') as f:
        json.dump(sequence_labels, f)

    return



class DEVA_run(object):
    def __init__(self, args) -> None:
        self.cfg = vars(args)
        self.cfg['enable_long_term'] = True
        self.cfg['model'] = os.path.join(BASE_DIR, 'Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth') 
        self.cfg['GROUNDING_DINO_CONFIG_PATH'] = os.path.join(BASE_DIR, 'Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py') 
        self.cfg['GROUNDING_DINO_CHECKPOINT_PATH'] = os.path.join(BASE_DIR, 'Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth') 
        self.cfg['SAM_ENCODER_VERSION'] = 'vit_b'
        self.cfg['SAM_CHECKPOINT_PATH'] = os.path.join(BASE_DIR, 'Tracking-Anything-with-DEVA/saves/sam_vit_b_01ec64.pth') 
        self.cfg['enable_long_term_count_usage'] = True
        self.cfg['max_num_objects'] = 100
        self.cfg['size'] = INT_SIZE
        self.cfg['amp'] = True
        self.cfg['chunk_size'] = 1
        self.cfg['num_voting_frames'] = 5
        self.cfg['detection_every'] = 2 # Changed to 1 because of low fps. Be careful, will return 0 masks 
        self.cfg['max_missed_detection_count'] = 20
        self.cfg['sam_variant'] = 'original'
        self.cfg['temporal_setting'] = 'semionline' # semionline usually works better; but online is faster for this demo
        self.cfg['pluralize'] = True
        self.cfg['suppress_small_objects'] = True
        self.cfg['SAM_NUM_POINTS_PER_SIDE'] = 20
        self.cfg['SAM_NUM_POINTS_PER_BATCH'] = 1
        self.cfg['mode'] = 'iou'
        CLASSES = ['paper', 'cardboard', 'metal', 'plastic', 'hand', 'background']
        self.cfg['prompt'] = '.'.join(CLASSES)
        self.voting_frames = [3, 5]
        self.detections_every = [2, 3, 5, 7]
        self.filter_styles = ["_post"]
        self.images = []
        self.image_source_dir = ""
        self.masks_dir = ""

    def runWithCheck(self, override:bool, frame_names:List[str], source_handler:interHandler, output_dir:str, suffix:str, sequence:str, reverse:bool, mask_handlers:List[interHandler] = []):
        target = path.join(output_dir+suffix, sequence)
        if(not override and path.exists(target)):
            return
        if(not path.exists(target)):
            print(f"{target} does not exist yet, creating")
        else:
            print(f"{target} being forced to rerun")
        singleRun(self.cfg, source_handler=source_handler, sequence=sequence, frame_names=frame_names, prefix=self.prefix, suffix=suffix, reversed=reverse, mask_handlers=mask_handlers)

    def postProcess(self, override:bool, output_dir:str, suffix:str, reverse:bool):
        return

    def run(self) -> None:
        for vote_frame in self.voting_frames:
            for detection_e in self.detections_every:
                self.prefix = str(vote_frame)+"_"+str(detection_e)
                output_dir = path.join(BASE_DIR, DATA_DIR, self.prefix, OUTPUT_DIR)
                labels_dir = self.masks_dir
                self.cfg['num_voting_frames']=vote_frame
                self.cfg['detection_every']=detection_e
                for image in self.images:
                    c_seq = image
                    frame_names = sorted(os.listdir(path.join(self.image_source_dir, c_seq)))
                    source_handler = interHandler(path.join(self.image_source_dir, c_seq), path.join(self.masks_dir, c_seq), path.join(labels_dir, c_seq), INT_SIZE)
                    self.runWithCheck(FORCE_FWD, frame_names=frame_names, source_handler=source_handler, output_dir=output_dir, suffix="_fwd", sequence=c_seq, reverse=False)
                    self.runWithCheck(FORCE_REV, frame_names=frame_names, source_handler=source_handler, output_dir=output_dir, suffix="_rev", sequence=c_seq, reverse=True)
                self.postProcess(FORCE_REV, output_dir=output_dir, suffix="_rev", reverse=True)

class auto_DEVA_run(DEVA_run):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.voting_frames = [5]
        self.detections_every = [5]
        #Manually set to the directory names you wish to process.
        self.images = ['CardboardOcclusions', 'Academic', 'Academic2']
        self.images = ['easy_occlusion', 'easy_occlusion2', 'easy_no_occlusions', 'b_occlusions_hard', 'b_occlusions_hard2']
        self.images = IMGS
        self.image_source_dir = path.join(BASE_DIR, RAW_IMAGE_DIR)
        self.masks_dir = path.join(BASE_DIR, DATA_DIR, "masks")


# for id2rgb
np.random.seed(42)

# default parameters
parser = ArgumentParser()
add_common_eval_args(parser)
add_ext_eval_args(parser)
add_text_default_args(parser)
parser.add_argument('--start', type=int,
                    help='First sequence to evaluate', default=1)
parser.add_argument('--end', type=int,
                    help='Last sequence to evaluate', default=18)

# load model and config
args = parser.parse_args()



run = auto_DEVA_run(args=args)
run.run()


  
  