from typing import List, Optional, Tuple, Dict
import torch
import torch.nn.functional as F
import torchvision
import os
from os import path
from PIL import Image, ImagePalette
import pycocotools.mask as mask_util
from threading import Thread
from queue import Queue
from dataclasses import dataclass
import copy

import numpy as np
import supervision as sv

from deva.utils.pano_utils import ID2RGBConverter
from deva.inference.object_manager import ObjectManager
from deva.inference.object_info import ObjectInfo

from skimage import measure
import cv2

class ResultSaver:
    def __init__(self,
                 output_root: str,
                 video_name: str,
                 *,
                 dataset: str,
                 object_manager: ObjectManager,
                 palette: Optional[ImagePalette.ImagePalette] = None,
                 just_final: bool = False):
        self.output_root = output_root
        self.video_name = video_name
        self.dataset = dataset.lower()
        self.palette = palette
        self.object_manager = object_manager

        self.need_remapping = False
        self.json_style = None
        self.output_postfix = None
        self.visualize = False
        self.id = 1000
        self.id2rgb_converter = ID2RGBConverter()
        self.just_final = just_final

        if self.dataset == 'vipseg':
            self.all_annotations = []
            self.video_json = {'video_id': video_name, 'annotations': self.all_annotations}
            self.need_remapping = True
            self.json_style = 'vipseg'
            self.output_postfix = 'pan_pred'
        elif self.dataset == 'burst':
            self.id2rgb_converter = ID2RGBConverter()
            self.need_remapping = True
            self.all_annotations = []
            dataset_name = path.dirname(video_name)
            seq_name = path.basename(video_name)
            self.video_json = {
                'dataset': dataset_name,
                'seq_name': seq_name,
                'segmentations': self.all_annotations
            }
            self.json_style = 'burst'
        elif self.dataset == 'unsup_davis17':
            self.need_remapping = True
        elif self.dataset == 'ref_davis':
            # nothing special is required
            pass
        elif self.dataset == 'demo':
            self.need_remapping = True
            self.all_annotations = []
            self.video_json = {'annotations': self.all_annotations}
            self.json_style = 'vipseg'
            self.visualize = True
            self.visualize_postfix = 'Visualizations'
            self.output_postfix = 'Annotations'
        elif self.dataset == 'gradio':
            # minimal mode, expect a cv2.VideoWriter to be assigned to self.writer asap
            self.writer = None
            self.need_remapping = True
            self.visualize = True
            self.json_style = 'coco'
            self.all_annotations = []
        elif self.dataset == 'coco':
            self.all_annotations = []
            self.video_json = {'video_id': video_name, 'annotations': self.all_annotations}
            self.need_remapping = True
            self.json_style = 'coco'
            self.output_postfix = 'pan_pred'
            self.all_annotations = []
            self.visualize = True
            self.visualize_postfix = 'Visualizations'
        else:
            raise NotImplementedError

        self.queue = Queue(maxsize=10)
        self.thread = Thread(target=save_result, args=(self.queue, ))
        self.thread.daemon = True
        self.thread.start()

    def save_mask(self,
                  prob: torch.Tensor,
                  frame_name: str,
                  need_resize: bool = False,
                  shape: Optional[Tuple[int, int]] = None,
                  save_the_mask: bool = True,
                  image_np: np.ndarray = None,
                  prompts: List[str] = None,
                  path_to_image: str = None,
                  save_ids: List[int] = None,
                  segments_override: List[Dict] = None,
                  frame_no: int = 0):

        if need_resize:
            prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,
                                                                                                 0]
        # Probability mask -> index mask
        mask = torch.argmax(prob, dim=0)
        if(segments_override is None):
            segments_override = self.object_manager.get_current_segments_info()
        # print(f"Saving {frame_no}", flush=True)
        args = ResultArgs(
            saver=self,
            mask=mask.cpu(),
            frame_name=frame_name,
            save_the_mask=save_the_mask,
            image_np=image_np,
            prompts=prompts,
            path_to_image=path_to_image,
            tmp_id_to_obj=copy.deepcopy(self.object_manager.tmp_id_to_obj),
            obj_to_tmp_id=copy.deepcopy(self.object_manager.obj_to_tmp_id),
            segments_info=copy.deepcopy(segments_override),
            save_ids=save_ids,
            frame_no = frame_no,
            just_final=self.just_final
        )

        self.queue.put(args)

    def end(self):
        self.queue.put(None)
        self.queue.join()
        self.thread.join()


@dataclass
class ResultArgs:
    saver: ResultSaver
    mask: torch.Tensor
    frame_name: str
    save_the_mask: bool
    image_np: np.ndarray
    prompts: List[str]
    path_to_image: str
    tmp_id_to_obj: Dict[int, ObjectInfo]
    obj_to_tmp_id: Dict[ObjectInfo, int]
    segments_info: List[Dict]
    save_ids: List[int]
    frame_no: int
    just_final: bool


def save_result(queue: Queue):
    # tracker = sv.ByteTrack()
    while True:
        args: ResultArgs = queue.get()
        if args is None:
            queue.task_done()
            break

        saver = args.saver
        mask = args.mask
        raw_mask = mask
        frame_name = args.frame_name
        save_the_mask = args.save_the_mask
        image_np = args.image_np
        prompts = args.prompts
        path_to_image = args.path_to_image
        tmp_id_to_obj = args.tmp_id_to_obj
        obj_to_tmp_id = args.obj_to_tmp_id
        segments_info = args.segments_info
        frame_no = args.frame_no
        just_final = args.just_final
        all_obj_ids = [k.id for k in obj_to_tmp_id]
        h = mask.shape[0]
        w = mask.shape[1]

        # remap indices
        if saver.need_remapping:
            new_mask = torch.zeros_like(mask)
            for tmp_id, obj in tmp_id_to_obj.items():
                new_mask[mask == tmp_id] = obj.id
            mask = new_mask

        # record output in the json file
        if saver.json_style == 'vipseg':
            for seg in segments_info:
                area = int((mask == seg['id']).sum())
                seg['area'] = area
            # filter out zero-area segments
            segments_info = [s for s in segments_info if s['area'] > 0]
            # append to video level storage
            this_annotation = {
                'file_name': frame_name[:-4] + '.jpg',
                'segments_info': segments_info,
            }
            saver.all_annotations.append(this_annotation)
        elif saver.json_style == 'burst':
            for seg in segments_info:
                seg['mask'] = mask == seg['id']
                seg['area'] = int(seg['mask'].sum())
                coco_mask = mask_util.encode(np.asfortranarray(seg['mask'].numpy()))
                coco_mask['counts'] = coco_mask['counts'].decode('utf-8')
                seg['rle_mask'] = coco_mask
            # filter out zero-area segments
            segments_info = [s for s in segments_info if s['area'] > 0]
            # append to video level storage
            this_annotation = {
                'file_name':
                frame_name[:-4] + '.jpg',
                'segmentations': [{
                    'id': seg['id'],
                    'score': seg['score'],
                    'rle': seg['rle_mask'],
                } for seg in segments_info],
            }
            saver.all_annotations.append(this_annotation)
        elif saver.json_style == 'coco':
            for seg in segments_info:
                
                seg['mask'] = mask == seg['id']
                seg['area'] = int(seg['mask'].sum())
                coco_mask = mask_util.encode(np.asfortranarray(seg['mask'].numpy()))
                contours = measure.find_contours(seg['mask'].numpy(), 0.5)
                bounding_box = mask_util.toBbox(coco_mask)
                segmentation_list = []
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                    segmentation_list.append(segmentation)
                seg['bbox'] = bounding_box.tolist()
                seg['segmentation'] = segmentation_list
                #Debug code
                # t_mask = seg['mask'].detach().numpy().astype(dtype=np.uint8)*255
                # for i in range(len(seg['segmentation'])):
                #     c_seg = seg['segmentation'][i]
                #     print(len(c_seg), c_seg)
                #     for j in range(int(len(c_seg)/2)):
                #         cv2.circle(t_mask, [int(c_seg[2*j]), int(c_seg[(2*j)+1])], 3, [0,0,255], -1)
                # cv2.imshow("mask", t_mask)
                # cv2.waitKey()
                if (just_final):
                    c_id = [tmp_id for tmp_id, obj in tmp_id_to_obj.items() if obj.id == seg['id']][0]
                    if(not tmp_id_to_obj[c_id].save):
                        seg['area'] = 0
                        mask[mask == seg['id']] = 0

            # filter out zero-area segments
            segments_info = [s for s in segments_info if s['area'] > 0]
            # append to video level storage
            this_annotation = [
                {
                    'image_id': frame_no,
                    'id': int(saver.id),
                    'score': float(seg['score']),
                    'area': float(seg['area']),
                    'category_id': (lambda x: x if x!=0 else 7)(int(seg['category_id'])),
                    'bbox': seg['bbox'],
                    'segmentation': seg['segmentation'],
                    'iscrowd': 0
                } for seg in segments_info]
            for i in range(len(this_annotation)):
                this_annotation[i]['id']+= i
                saver.all_annotations.append(this_annotation[i])
            saver.id += len(this_annotation)
        elif saver.visualize:
            # if we are visualizing, we need to preprocess segment info
            for seg in segments_info:
                area = int((mask == seg['id']).sum())
                seg['area'] = area
            # filter out zero-area segments
            segments_info = [s for s in segments_info if s['area'] > 0]

        # save the mask to disk
        if save_the_mask:
            
            # else:
            out_mask = raw_mask.numpy().astype(np.uint8)
            out_img = Image.fromarray(out_mask)
            if saver.palette is not None:
                out_img.putpalette(saver.palette)

            if saver.dataset != 'gradio':
                # find a place to save the mask
                if saver.output_postfix is not None:
                    this_out_path = path.join(saver.output_root, saver.output_postfix)
                else:
                    this_out_path = saver.output_root
                if saver.video_name is not None:
                    this_out_path = path.join(this_out_path, saver.video_name)

                os.makedirs(this_out_path, exist_ok=True)
                out_img.save(path.join(this_out_path, frame_name[:-4] + '.png'))
            if saver.object_manager.use_long_id:
                out_mask = mask.numpy().astype(np.uint32)
                rgb_mask = np.zeros((*out_mask.shape[-2:], 3), dtype=np.uint8)
                blue = np.array([255, 0, 0], dtype=np.uint8)
                for id in all_obj_ids:
                    colored_mask = saver.id2rgb_converter._id_to_rgb(id)
                    obj_mask = (out_mask == id)
                    rgb_mask[obj_mask] = blue

            if saver.visualize and saver.object_manager.use_long_id:
                if image_np is None:
                    if path_to_image is not None:
                        image_np = np.array(Image.open(path_to_image))
                    else:
                        raise ValueError('Cannot visualize without image_np or path_to_image')
                alpha = (out_mask == 0).astype(np.float32) * 0.5 + 0.5
                alpha = alpha[:, :, None]
                blend = (image_np * alpha + rgb_mask * (1 - alpha)).astype(np.uint8)

                if prompts is not None:
                    # draw bounding boxes for the prompts
                    all_masks = []
                    labels = []
                    all_cat_ids = []
                    all_scores = []
                    check_ids = []
                    if(args.save_ids is not None):
                        check_ids = args.save_ids

                    #Updated visualization for our use
                    for seg in segments_info:
                        color = saver.id2rgb_converter._id_to_rgb(seg['id'])
                        color = (255, 0, 0)
                        if(not just_final):
                            c_cen = seg['centroids']
                            l = len(c_cen)

                            for i in range(l-1):
                                cv2.line(blend, [int(w*c_cen[i][0]), int(h*c_cen[i][1])], [int(w*c_cen[i+1][0]), int(h*c_cen[i+1][1])], (0,0,0), 4)
                                cv2.line(blend, [int(w*c_cen[i][0]), int(h*c_cen[i][1])], [int(w*c_cen[i+1][0]), int(h*c_cen[i+1][1])], color, 2)
                            cv2.circle(blend, [int(w*c_cen[-1][0]), int(h*c_cen[-1][1])], 5, (0,0,0), -1)
                            cv2.circle(blend, [int(w*c_cen[-1][0]), int(h*c_cen[-1][1])], 3, color, -1)
                        
                        
                        if(seg['save']):
                            temp = np.zeros(mask.shape, dtype=np.uint8)
                            temp[mask==seg['id']] = 255
                            contours, _ = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                            blend = cv2.drawContours(blend, contours, -1, color, 2)
                            if(not just_final):
                                r5 = [-1,-1]
                                if(2 <= len(c_cen)):
                                    r5 = 100*np.mean(np.diff(np.array(c_cen), axis=0)[-10:], axis=0)
                                blend = cv2.putText(blend, f"[{r5[0]:.2f}, {r5[1]:.2f}]", [int(w*c_cen[-1][0]), int(h*c_cen[-1][1])], cv2.FONT_HERSHEY_PLAIN, 1, [0,0,0], 2)
                                
                    if len(all_masks) > 0:
                        masks = torch.stack(all_masks, dim=0)
                        xyxy = torchvision.ops.masks_to_boxes(masks)
                        xyxy = xyxy.numpy()

                        detections = sv.Detections(xyxy,
                                                   confidence=np.array(all_scores),
                                                   class_id=np.array(all_cat_ids))
                        annotator = sv.BoxAnnotator()
                        blend = annotator.annotate(scene=blend,
                                                   detections=detections,
                                                   labels=labels)
                        

                if saver.dataset != 'gradio':
                    # find a place to save the visualization
                    if saver.visualize_postfix is not None:
                        this_out_path = path.join(saver.output_root, saver.visualize_postfix)
                    else:
                        this_out_path = saver.output_root
                    if saver.video_name is not None:
                        # this_out_path = path.join(this_out_path, saver.video_name)
                        p, sequence = path.split(saver.video_name)
                        _, parent = path.split(p)
                        this_out_path = path.join(this_out_path, parent, sequence)

                    os.makedirs(this_out_path, exist_ok=True)
                    Image.fromarray(blend).save(path.join(this_out_path, frame_name[:-4] + '.jpg'))
                else:
                    saver.writer.write(blend[:, :, ::-1])

        queue.task_done()