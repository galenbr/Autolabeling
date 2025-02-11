from PIL import Image
import os
from os import path
import sys
import numpy as np
import tqdm

from deva.utils.palette import davis_palette


def limit_max_id(input_path, output_path, max_num_objects=20, reverse=False):
    videos = sorted(os.listdir(input_path))
    for video in tqdm.tqdm(videos):
        existing_objects = []

        video_path = path.join(input_path, video)
        frames = sorted(os.listdir(video_path), reverse=reverse)

        # determine the objects to keep
        for frame in frames:
            mask = Image.open(path.join(video_path, frame))
            mask = np.array(mask).astype(np.int32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
            labels = np.unique(mask)
            labels = labels[labels != 0]
            labels_area = [np.sum(mask == label) for label in labels]

            labels_sorted_by_area = [x for _, x in sorted(zip(labels_area, labels), reverse=True)]
            if len(labels_sorted_by_area) + len(existing_objects) <= max_num_objects:
                existing_objects += labels_sorted_by_area
            else:
                existing_objects += labels_sorted_by_area[:max_num_objects - len(existing_objects)]

            if len(existing_objects) == max_num_objects:
                break

        assert len(existing_objects) <= max_num_objects

        # remove the objects that are not in the existing_objects list
        for frame in frames:
            mask = Image.open(path.join(video_path, frame))
            mask = np.array(mask).astype(np.int32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
            labels = np.unique(mask)
            labels = labels[labels != 0]

            new_mask = np.zeros_like(mask, dtype=np.uint8)
            for new_idx, label in enumerate(existing_objects):
                new_mask[mask == label] = new_idx + 1

            mask = Image.fromarray(new_mask)
            mask.putpalette(davis_palette)
            os.makedirs(path.join(output_path, video), exist_ok=True)
            mask.save(path.join(output_path, video, frame))

def smart_limit_max_id(input_path, output_path, max_num_objects=20, reverse=False):
    videos = sorted(os.listdir(input_path))
    for video in tqdm.tqdm(videos):
        final_objects = []
        existing_objects = {}

        video_path = path.join(input_path, video)
        frames = sorted(os.listdir(video_path), reverse=reverse)

        # determine the objects to keep
        for frame in frames:
            mask = Image.open(path.join(video_path, frame))
            mask = np.array(mask).astype(np.int32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
            labels = np.unique(mask)
            labels = labels[labels != 0]
            labels_area = [np.sum(mask == label) for label in labels]

            # labels_sorted_by_area = [(a, x) for a, x in sorted(zip(labels_area, labels), reverse=True)]
            for label, area in zip(labels, labels_area):
                #https://stackoverflow.com/questions/24801729/how-to-add-to-a-dictionary-value-or-create-if-not-exists
                existing_objects[label] = (existing_objects.get(label,(0,0))[0]+1, existing_objects.get(label,(0,0))[1]+area)

        ordered_objects = sorted(existing_objects.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
        final_objects = [x for (x, _) in ordered_objects]
        final_objects = final_objects[:min(max_num_objects, len(final_objects))]

        assert len(final_objects) <= max_num_objects

        # remove the objects that are not in the final_objects list
        for frame in frames:
            mask = Image.open(path.join(video_path, frame))
            mask = np.array(mask).astype(np.int32)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 2]
            labels = np.unique(mask)
            labels = labels[labels != 0]

            new_mask = np.zeros_like(mask, dtype=np.uint8)
            for new_idx, label in enumerate(final_objects):
                new_mask[mask == label] = new_idx + 1

            mask = Image.fromarray(new_mask)
            mask.putpalette(davis_palette)
            os.makedirs(path.join(output_path, video), exist_ok=True)
            mask.save(path.join(output_path, video, frame))

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    limit_max_id(input_path, output_path)