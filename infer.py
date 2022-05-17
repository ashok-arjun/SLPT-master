import argparse
import os

from Config import cfg
from Config import update_config

from utils import create_logger
from SLPT import Sparse_alignment_network
from Dataloader import CustomDataset

import torch
import numpy as np
import pprint
import cv2
import os
from tqdm import tqdm
from pathlib import Path
import random
import shutil

import torchvision.transforms as transforms

import warnings
warnings.simplefilter("ignore", UserWarning)


def parse_args():
  parser = argparse.ArgumentParser(description='Train Sparse Facial Network')

  # landmark_detector
  parser.add_argument('--modelDir', help='model directory', type=str, default='./Weight')
  parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='WFLW_6_layer.pth')
  parser.add_argument('--logDir', help='log directory', type=str, default='./log')
  parser.add_argument('--dataDir', help='data directory', type=str, default='./')
  parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)
  parser.add_argument('--batchSize', help='batch size', type=int, default=1)
  parser.add_argument('--targetDir', type=str, required=True)
  parser.add_argument('--numWorkers', type=int, required=8)
  parser.add_argument('--demo', action='store_true')
  parser.add_argument('--demo_dir', type=str, required=True)
  parser.add_argument('--glob', type=str, default='*/*.jpg')

  args = parser.parse_args()

  return args


def calcuate_loss(name, pred, gt, trans):

  pred = (pred - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)

  if name == 'WFLW':
      norm = np.linalg.norm(gt[60, :] - gt[72, :])
  elif name == '300W':
      norm = np.linalg.norm(gt[36, :] - gt[45, :])
  elif name == 'COFW':
      norm = np.linalg.norm(gt[17, :] - gt[16, :])
  else:
      raise ValueError('Wrong Dataset')

  error_real = np.mean(np.linalg.norm((pred - gt), axis=1) / norm)

  return error_real


def main_function():

  args = parse_args()
  update_config(cfg, args)
  pprint.pprint(args)

  # create logger
  logger = create_logger(cfg)
  logger.info(pprint.pformat(args))
  logger.info(cfg)

  torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
  torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
  torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

  # NUM_POINT is 98, OUT_DIM is 256, NUM_DECODER is 6
  model = Sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
                                  cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                  cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                  cfg.TRANSFORMER.FEED_DIM, cfg.WFLW.INITIAL_PATH, cfg)

  model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  )

  valid_dataset = CustomDataset(
      cfg, args.dataDir,
      transforms.Compose([
          transforms.ToTensor(),
          normalize,
      ]),
      givenGlob = args.glob
  )

  valid_loader = torch.utils.data.DataLoader(
      valid_dataset,
      batch_size = args.batchSize,
      shuffle=False,
      num_workers=args.numWorkers,
      pin_memory=cfg.PIN_MEMORY
  )

  checkpoint_file = os.path.join(args.modelDir, args.checkpoint)
  checkpoint = torch.load(checkpoint_file)

  pretrained_dict = {k: v for k, v in checkpoint.items()
                      if k in model.module.state_dict().keys()}

  model.module.load_state_dict(pretrained_dict)
  model.eval()

  paths = []
  outputs = []

  print("Total number of samples:", len(valid_dataset))
  print("Batch Size:", args.batchSize, "Total number of batches:", len(valid_loader))

  with torch.no_grad():
    for i, (input, path, BBox) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
      outputs_initial = model(input.cuda())
      output = outputs_initial[2][:, -1, :, :].cpu().numpy()
      paths.append(path)
      outputs.append(output)

  print("Finished")

  outputs = np.concatenate(outputs, axis=0)
  paths = np.concatenate(paths, axis=0)

  print("Writing landmarks to disk...")
  for i, (path, output) in enumerate(zip(paths, outputs)):
      imgname = Path(path).name
      imgNameRoot, imgNameExtension = os.path.basename(path).split('.')

      data_root_parts = args.dataDir.split("/")
      vid_parts = path.split("/")
      remainingPath = "/".join(vid_parts[len(data_root_parts):-1])

      targetDir = os.path.join(args.targetDir, remainingPath)
      os.makedirs(targetDir, exist_ok=True)

      print("Saving landmark %i to %s" % (i, targetDir))
      file = open(os.path.join(targetDir, imgNameRoot+".txt"), "w")
            
      for j, row in enumerate(output):
        file.write(str(row[0])+" "+str(row[1]))
        if j != len(output) - 1:
          file.write(" ")

      file.close()
  print("Done.")

  if args.demo_dir in os.listdir("./"):
    shutil.rmtree(args.demo_dir)
  os.makedirs(args.demo_dir, exist_ok=True)

  if args.demo and len(paths):
    print("Writing demo images...")

    indexes = list(range(len(paths)))

    os.makedirs(args.demo_dir, exist_ok=True)

    for INDEX in indexes:
      print("Processing %d/%d" % (INDEX, len(indexes)))
      PATH = paths[INDEX]
      img = cv2.imread(PATH)
      img = cv2.resize(img, (cfg.MODEL.IMG_SIZE, cfg.MODEL.IMG_SIZE), interpolation = cv2.INTER_AREA)

      imgname = Path(PATH).name
      imgNameRoot, imgNameExtension = os.path.basename(PATH).split('.')
      fullRoot = "-".join(PATH.split("/")[:-1])

      cv2.imwrite(os.path.join(args.demo_dir, fullRoot, imgname), img)
      h, w, c = img.shape

      x_y_coordinates = outputs[INDEX]

      for x, y in x_y_coordinates:
        img = cv2.circle(img, (round(x*h), round(y*w)), radius=2, color=(255, 0, 0), thickness=-1)

      os.makedirs(os.path.join(args.demo_dir, fullRoot), exist_ok=True)

      cv2.imwrite(os.path.join(args.demo_dir, fullRoot, imgNameRoot + "-landmarked" + "." + imgNameExtension), img)
    print("Done.")


if __name__ == '__main__':
    main_function()

