from typing import Optional
import numpy as np
from scipy import stats
from deva.utils.pano_utils import id_to_rgb
from typing import Union, List

def avg_centroids(c1: List[float], c2: List[float]):
    if(-1 == c1[0] and -1 != c2[0]):
        return c2
    if(-1 == c2[0] and -1 != c1[0]):
        return c1
    x = c1[0] + c2[0]
    y = c1[1] + c2[1]
    return [x/2, y/2]

def avg_lists(l1: List[List[float]], l2: List[List[float]]):
    len_1 = len(l1)
    len_2 = len(l2)
    m_range = min(len_1, len_2)
    ret = []
    if(len_1 > m_range):
        ret = l1[:-m_range]
    if(len_2 > m_range):
        ret = l2[:-m_range]
    for i in range(m_range, 0, -1):
        ret.append(avg_centroids(l1[-i], l2[-i]))
    return ret

class ObjectInfo:
    """
    Stores meta information for an object
    """
    def __init__(self,
                 id: int,
                 category_id: Optional[int] = None,
                 isthing: Optional[bool] = None,
                 score: Optional[float] = None):
        self.id = id
        self.category_ids = [category_id]
        self.scores = [score]
        self.isthing = isthing
        self.poke_count = 0  # number of detections since last this object was last seen
        self.centroids = []
        self.max_disp = 0
        self.recent_disp = [1,1]
        self.save = False

    #Tracks the object centroid and performs the first "removal" check
    def set_centroid(self, x, y):
        #If it's a 0 detection, just reuse the last centroid. 
        if(-1 == x and -1 == y):
            self.centroids.append(self.get_centroid())
        else:
            self.centroids.append([x, y])
        if(2 <= len(self.centroids)):
            self.max_disp = max(self.max_disp, np.linalg.norm(np.array(self.centroids[-2])-np.array(self.centroids[-1])))
        if(30 <= len(self.centroids)):
            recent = np.array(self.centroids[-30:])
            self.recent_disp = np.max(recent, axis=0)-np.min(recent, axis=0)
        if (not self.save) and (20 >= len(self.centroids)) and (5 <= len(self.centroids)):
            movement = 100*np.mean(np.diff(self.centroids[-5:], axis=0), axis=0)
            self.save = (-1.5 > movement[1]) and (.1 < self.centroids[-1][0])

    def poke(self) -> None:
        self.poke_count += 1

    def unpoke(self) -> None:
        self.poke_count = 0

    def merge(self, other) -> None:
        self.category_ids.extend(other.category_ids)
        self.scores.extend(other.scores)
        
    def vote_category_id(self) -> Optional[int]:
        category_ids = [c for c in self.category_ids if c is not None]
        if len(category_ids) == 0:
            return None
        else:
            return int(stats.mode(category_ids, keepdims=False)[0])

    def vote_score(self) -> Optional[float]:
        scores = [c for c in self.scores if c is not None]
        if len(scores) == 0:
            return None
        else:
            return float(np.mean(scores))
        
    def get_all_centroids(self) -> List[List[float]]:
        return self.centroids
    
    def get_centroid(self) -> List[float]:
        if(0 >= len(self.centroids)):
            return [-1, -1]
        return self.centroids[-1]

    def get_rgb(self) -> np.ndarray:
        # this is valid for panoptic segmentation-style id only (0~255**3)
        return id_to_rgb(self.id)

    def copy_meta_info(self, other) -> None:
        self.category_ids = other.category_ids
        self.scores = other.scores
        self.isthing = other.isthing
        self.centroids = other.centroids
        self.save = other.save or self.save

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f'(ID: {self.id}, cat: {self.category_ids}, isthing: {self.isthing}, score: {self.scores}, centroids: ({self.centroids}), save: {self.save})'
