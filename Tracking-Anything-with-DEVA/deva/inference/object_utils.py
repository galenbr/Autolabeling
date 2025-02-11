from typing import List, Optional

import torch
from deva.inference.object_info import ObjectInfo
from deva.utils.pano_utils import vipseg_cat_to_isthing

#Simplified conditions for our use case
def convert_json_dict_to_objects_info(mask: torch.Tensor,
                                      segments_info: Optional[List],
                                      dataset: str = None) -> List[ObjectInfo]:
    """
    Convert a json dict to a list of object info
    If segments_info is None, we use the unique elements in mask to construct the list
    Otherwise mask is ignored
    """
    if segments_info is not None:
        #Original code
        # output = [
        #     ObjectInfo(
        #         id=segment['id'],
        #         category_id=segment.get('category_id'),
        #         isthing=vipseg_cat_to_isthing[segment.get('category_id')]
        #         if dataset == 'vipseg' and segment.get('category_id') is not None else None,
        #         score=float(segment['score']) if
        #         ((dataset == 'burst' or dataset == 'demo' or dataset == 'coco') and 'score' in segment) else None)
        #     for segment in segments_info
        # ]
        output = []
        i = 0
        for segment in segments_info:
            o = ObjectInfo(
                id=i+1,
                category_id=segment.get('category_id'),
                isthing=vipseg_cat_to_isthing[segment.get('category_id')]
                if dataset == 'vipseg' else None,
                score=float(segment['score']) if
                ((dataset == 'burst' or dataset == 'demo' or dataset == 'coco') and 'score' in segment) else None)
            i+=1
            output.append(o)
    else:
        # use the mask
        labels = torch.unique(mask)
        labels = labels[labels != 0]
        output = [ObjectInfo(l.item()) for l in labels]

    return output
