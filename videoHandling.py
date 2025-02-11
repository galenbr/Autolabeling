import numpy as np
import cv2 as cv
from config import *
import os

def fixSuffix(string: str) -> str:
  if(4 > len(string)):
     raise Exception(f"Video file name {string} less than 4 characters long.")
  if('.mp4' != string[-4:]):
    string += '.mp4'
  return string

def vid2Seq(video, out_dir):
  vidcap = cv.VideoCapture(video)
  os.makedirs(out_dir, exist_ok=True)
  i = 0
  while vidcap.isOpened():
      success, image = vidcap.read()
      if success:
          cv.imwrite(os.path.join(out_dir, f"{i:05}.png"),image)
          i+=1
      else:
          break
  
  return True

def seq2Vid(sequence, name):
  fourcc = cv.VideoWriter_fourcc(*'mp4v')
  img = cv.imread(os.path.join(sequence, "00000.jpg"))
  #Debug code
  # print(img.shape)
  # out = cv.VideoWriter(name+'_rev.mp4', fourcc, 5.0, (img.shape[1], img.shape[0]))
  # for i in range(len(os.listdir(sequence))-1,-1,-1):
  #   print(os.path.join(sequence, f"{i:05}.jpg"))
  #   img = cv.imread(os.path.join(sequence, f"{i:05}.jpg"))
  #   out.write(img)
  # out.release()
  out = cv.VideoWriter(os.path.join("/home/rbe07/Documents/Google/data/sequences/Videos", name+'.mp4'), fourcc, 30.0, (img.shape[1], img.shape[0]))
  for i in range(int(len(os.listdir(sequence)))):
    img = cv.imread(os.path.join(sequence, f"{i:05}.jpg"))
    out.write(img)
  out.release()

def revVid(video, name):
  name = fixSuffix(name)
  vidcap = cv.VideoCapture(video)
  imgs = []
  while vidcap.isOpened():
      success, image = vidcap.read()
      if success:
          imgs.append(image)
      else:
          break
  print("Done")
  vidcap.release()
  fourcc = cv.VideoWriter_fourcc(*'MJPG')
  out = cv.VideoWriter(name, fourcc, 5.0, (imgs[0].shape[1], imgs[0].shape[0]))
  for i in range(len(imgs)-1,-1,-1):
    out.write(imgs[i])
  print(len(imgs))
  out.release()
  
def combineVideos(videos, name, offset=0):
  fixSuffix(name)
  vidcaps = [cv.VideoCapture(v) for v in videos]
  c = 0
  fourcc = cv.VideoWriter_fourcc(*'MJPG')
  out = cv.VideoWriter(name, fourcc, 60.0, (int(len(vidcaps)*vidcaps[0].get(cv.CAP_PROP_FRAME_WIDTH)), int(vidcaps[0].get(cv.CAP_PROP_FRAME_HEIGHT))))
  while(c < offset):
    success, image = vidcaps[0].read()
    c += 1
  while vidcaps[0].isOpened():
      cont = True
      imgs = []
      for vidcap in vidcaps:
        
        success, image = vidcap.read()
        cont = cont and success
        imgs.append(image)
      if cont:
          img = cv.hconcat(imgs)
          out.write(img)
      else:
          break
  for vidcap in vidcaps:
    vidcap.release()
  out.release()



if __name__ == '__main__':
  # if(not os.path.isfile("/home/rbe07/Documents/Google/data/sequences/Videos/CO_both.mp4")):
  #   seq2Vid("/home/rbe07/Documents/Google/data/5_5/DEVA_output/Visualizations/DEVA_output_fwd/CardboardOcclusions", "CO_fwd")
  #   seq2Vid("/home/rbe07/Documents/Google/data/5_5/DEVA_output/Visualizations/DEVA_output_rev/CardboardOcclusions", "CO_rev")
  #   seq2Vid("/home/rbe07/Documents/Google/data/5_5/DEVA_output/Visualizations/DEVA_output_both_4_post/CardboardOcclusions", "CO_both")
  # combineVideos(["/home/rbe07/Documents/Google/data/sequences/Videos/CO_fwd.mp4", "/home/rbe07/Documents/Google/data/sequences/Videos/CO_rev.mp4", "/home/rbe07/Documents/Google/data/sequences/Videos/CO_both.mp4"], "/home/rbe07/Documents/Google/data/sequences/Videos/CO_comp.mp4")
  seq2Vid("/home/rbe07/Documents/Google/data/5_5/DEVA_output/Visualizations/DEVA_output_rev/CardboardOcclusions", "CO_ours_2")
  # seq2Vid("/home/rbe07/Documents/Google/data/5_5/DEVA_output/Visualizations/DEVA_output_rev/CardboardOcclusionsCorrected", "CO_hand")
  seq2Vid("/home/rbe07/Documents/DEVA_rep/Tracking-Anything-with-DEVA/example/output/Visualizations", "CO_theirs_2")
  # seq2Vid("/home/rbe07/Documents/DEVA_rep/Tracking-Anything-with-DEVA/example/output/Visualizations", "CO_post")
  
  # seq2Vid("/home/rbe07/Documents/Google/data/sequences", "med_paper")
  # revVid("/home/rbe07/Downloads/s2_rev.mp4", "/home/rbe07/Downloads/s2_rev2")
  # combineVideos("/home/rbe07/Downloads/s2.mp4", "/home/rbe07/Downloads/s2_rev2.mp4", "/home/rbe07/Downloads/s2_both")
  pass