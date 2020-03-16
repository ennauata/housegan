#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, random, math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils
import glob
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageFont, ImageColor
import matplotlib.pyplot as plt
import random
from pygraphviz import *
import cv2
from torchvision.utils import save_image

EXP_ID = random.randint(0, 1000000)

#     labelMap['living_room'] = 1
#     labelMap['kitchen'] = 2
#     labelMap['bedroom'] = 3
#     labelMap['bathroom'] = 4
#     labelMap['restroom'] = 4
#     labelMap['washing_room'] = 4    
#     labelMap['office'] = 3
#     labelMap['closet'] = 6
#     labelMap['balcony'] = 7
#     labelMap['corridor'] = 8
#     labelMap['dining_room'] = 9
#     labelMap['laundry_room'] = 10
#     labelMap['PS'] = 10   
ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8,
              "dining_room": 9, "laundry_room": 10}

CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x

ID_COLOR = {1: 'brown', 2: 'magenta', 3: 'orange', 4: 'gray', 5: 'red', 6: 'blue', 7: 'cyan', 8: 'green', 9: 'salmon', 10: 'yellow'}
NUM_WALL_CORNERS = 13
NUM_CORNERS = 21
#CORNER_RANGES = {'wall': (0, 13), 'opening': (13, 17), 'icon': (17, 21)}

NUM_ICONS = 7
NUM_ROOMS = 10
POINT_ORIENTATIONS = [[(2, ), (3, ), (0, ), (1, )], [(0, 3), (0, 1), (1, 2), (2, 3)], [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)], [(0, 1, 2, 3)]]

class ColorPalette:
    def __init__(self, numColors):
        #np.random.seed(2)
        #self.colorMap = np.random.randint(255, size = (numColors, 3))
        #self.colorMap[0] = 0

        
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],                                   
                                  [255, 255, 0],                                  
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],                                  
                                  [0, 100, 100],
                                  [0, 255, 128],                                  
                                  [0, 128, 255],
                                  [255, 0, 128],                                  
                                  [128, 0, 255],
                                  [255, 128, 0],                                  
                                  [128, 255, 0],                                                                    
        ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.random.randint(255, size = (numColors, 3))
            pass
        
        return

    def getColorMap(self):
        return self.colorMap
    
    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass
        return

def isManhattan(line, gap=3):
    return min(abs(line[0][0] - line[1][0]), abs(line[0][1] - line[1][1])) < gap

def calcLineDim(points, line):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    if abs(point_2[0] - point_1[0]) > abs(point_2[1] - point_1[1]):
        lineDim = 0
    else:
        lineDim = 1
        pass
    return lineDim

def calcLineDirection(line, gap=3):
    return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))


## Draw segmentation image. The input could be either HxW or HxWxC
def drawSegmentationImage(segmentations, numColors=42, blackIndex=-1, blackThreshold=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        if blackThreshold > 0:
            segmentations = np.concatenate([segmentations, np.ones((segmentations.shape[0], segmentations.shape[1], 1)) * blackThreshold], axis=2)
            blackIndex = segmentations.shape[2] - 1
            pass

        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        #segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass

    segmentation = segmentation.astype(np.int32)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))


def drawWallMask(walls, width, height, thickness=3, indexed=False):
    if indexed:
        wallMask = np.full((height, width), -1, dtype=np.int32)
        for wallIndex, wall in enumerate(walls):
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=wallIndex, thickness=thickness)
            continue
    else:
        wallMask = np.zeros((height, width), dtype=np.int32)
        for wall in walls:
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=1, thickness=thickness)
            continue
        wallMask = wallMask.astype(np.bool)
        pass
    return wallMask


def extractCornersFromHeatmaps(heatmaps, heatmapThreshold=0.5, numPixelsThreshold=5, returnRanges=True):
    """Extract corners from heatmaps"""
    from skimage import measure 
    print(heatmaps.shape)
    
    heatmaps = (heatmaps > heatmapThreshold).astype(np.float32)
    orientationPoints = []
    #kernel = np.ones((3, 3), np.float32)
    for heatmapIndex in range(0, heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, heatmapIndex]
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min() + 1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            if ys.shape[0] <= numPixelsThreshold:
                continue
            #print(heatmapIndex, xs.shape, ys.shape, componentIndex)
            if returnRanges:
                points.append(((xs.mean(), ys.mean()), (xs.min(), ys.min()), (xs.max(), ys.max())))
            else:
                points.append((xs.mean(), ys.mean()))
                pass
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

def extractCornersFromSegmentation(segmentation, cornerTypeRange=[0, 13]):
    """Extract corners from segmentation"""
    from skimage import measure
    orientationPoints = []
    for heatmapIndex in range(cornerTypeRange[0], cornerTypeRange[1]):
        heatmap = segmentation == heatmapIndex
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in range(components.min()+1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            points.append((xs.mean(), ys.mean()))
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

def getOrientationRanges(width, height):
    orientationRanges = [[width, 0, 0, 0], [width, height, width, 0], [width, height, 0, height], [0, height, 0, 0]]
    return orientationRanges

def getIconNames():
    iconNames = []
    iconLabelMap = getIconLabelMap()
    for iconName, _ in iconLabelMap.items():
        iconNames.append(iconName)
        continue
    return iconNames

def getIconLabelMap():
    labelMap = {}
    labelMap['bathtub'] = 1
    labelMap['cooking_counter'] = 2
    labelMap['toilet'] = 3
    labelMap['entrance'] = 4
    labelMap['washing_basin'] = 5
    labelMap['special'] = 6
    labelMap['stairs'] = 7
    labelMap['door'] = 8
    return labelMap


def drawPoints(filename, width, height, points, backgroundImage=None, pointSize=5, pointColor=None):
  colorMap = ColorPalette(NUM_CORNERS).getColorMap()
  if np.all(np.equal(backgroundImage, None)):
    image = np.zeros((height, width, 3), np.uint8)
  else:
    if backgroundImage.ndim == 2:
      image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 3])
    else:
      image = backgroundImage
      pass
  pass
  no_point_color = pointColor is None
  for point in points:
    if no_point_color:
        pointColor = colorMap[point[2] * 4 + point[3]]
        pass
    #print('used', pointColor)
    #print('color', point[2] , point[3])
    image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width)] = pointColor
    continue

  if filename != '':
    cv2.imwrite(filename, image)
    return
  else:
    return image

def drawPointsSeparately(path, width, height, points, backgroundImage=None, pointSize=5):
  if np.all(np.equal(backgroundImage, None)):
    image = np.zeros((height, width, 13), np.uint8)
  else:
    image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 13])
    pass

  for point in points:
    image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width), int(point[2] * 4 + point[3])] = 255
    continue
  for channel in range(13):
    cv2.imwrite(path + '_' + str(channel) + '.png', image[:, :, channel])
    continue
  return

def drawLineMask(width, height, points, lines, lineWidth = 5, backgroundImage = None):
  lineMask = np.zeros((height, width))

  for lineIndex, line in enumerate(lines):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    direction = calcLineDirectionPoints(points, line)

    fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
    minValue = int(min(point_1[direction], point_2[direction]))
    maxValue = int(max(point_1[direction], point_2[direction]))
    if direction == 0:
      lineMask[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1] = 1
    else:
      lineMask[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width)] = 1
      pass
    continue
  return lineMask



def drawLines(filename, width, height, points, lines, lineLabels = [], backgroundImage = None, lineWidth = 5, lineColor = None):
  colorMap = ColorPalette(len(lines)).getColorMap()
  if backgroundImage is None:
    image = np.ones((height, width, 3), np.uint8) * 0
  else:
    if backgroundImage.ndim == 2:
      image = np.stack([backgroundImage, backgroundImage, backgroundImage], axis=2)
    else:
      image = backgroundImage
      pass
    pass

  for lineIndex, line in enumerate(lines):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    direction = calcLineDirectionPoints(points, line)


    fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
    minValue = int(round(min(point_1[direction], point_2[direction])))
    maxValue = int(round(max(point_1[direction], point_2[direction])))
    if len(lineLabels) == 0:
      if np.any(lineColor == None):
        lineColor = np.random.rand(3) * 255
        pass
      if direction == 0:
        image[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1, :] = lineColor
      else:
        image[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width), :] = lineColor
    else:
      labels = lineLabels[lineIndex]
      isExterior = False
      if direction == 0:
        for c in range(3):
          image[max(fixedValue - lineWidth, 0):min(fixedValue, height), minValue:maxValue, c] = colorMap[labels[0]][c]
          image[max(fixedValue, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue, c] = colorMap[labels[1]][c]
          continue
      else:
        for c in range(3):
          image[minValue:maxValue, max(fixedValue - lineWidth, 0):min(fixedValue, width), c] = colorMap[labels[1]][c]
          image[minValue:maxValue, max(fixedValue, 0):min(fixedValue + lineWidth + 1, width), c] = colorMap[labels[0]][c]
          continue
        pass
      pass
    continue

  if filename == '':
    return image
  else:
    cv2.imwrite(filename, image)


def drawRectangles(filename, width, height, points, rectangles, labels, lineWidth = 2, backgroundImage = None, rectangleColor = None):
  colorMap = ColorPalette(NUM_ICONS).getColorMap()
  if backgroundImage is None:
    image = np.ones((height, width, 3), np.uint8) * 0
  else:
    image = backgroundImage
    pass

  for rectangleIndex, rectangle in enumerate(rectangles):
    point_1 = points[rectangle[0]]
    point_2 = points[rectangle[1]]
    point_3 = points[rectangle[2]]
    point_4 = points[rectangle[3]]


    if len(labels) == 0:
      if rectangleColor is None:
        color = np.random.rand(3) * 255
      else:
        color = rectangleColor
    else:
      color = colorMap[labels[rectangleIndex]]
      pass

    x_1 = int(round((point_1[0] + point_3[0]) / 2))
    x_2 = int(round((point_2[0] + point_4[0]) / 2))
    y_1 = int(round((point_1[1] + point_2[1]) / 2))
    y_2 = int(round((point_3[1] + point_4[1]) / 2))

    cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color=tuple(color.tolist()), thickness = 2)
    continue

  if filename == '':
    return image
  else:
    cv2.imwrite(filename, image)
    pass

def pointDistance(point_1, point_2):
    #return np.sqrt(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2))
    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))

def calcLineDirectionPoints(points, line):
  point_1 = points[line[0]]
  point_2 = points[line[1]]
  if isinstance(point_1[0], tuple):
      point_1 = point_1[0]
      pass
  if isinstance(point_2[0], tuple):
      point_2 = point_2[0]
      pass
  return calcLineDirection((point_1, point_2))

def open_png(im_path, im_size=512):
	
	# open graph image
	png = Image.open(im_path)
	im = Image.new("RGB", png.size, (255, 255, 255))
	im.paste(png, mask=png.split()[3])
	w, h = im.size
	
    # pad graph images
	a = h/w
	if w > h:
		n_w = im_size
		n_h = int(a*n_w)
	else:
		n_h = im_size
		n_w = int(n_h/a)
	im = im.resize((n_w, n_h))
	delta_w = im_size - n_w
	delta_h = im_size - n_h
	padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
	im = ImageOps.expand(im, padding, fill='white')
	im_arr = np.array(im)
	
	return im_arr

def draw_graph(nds, eds, shift, im_size=128):

    # Create graph
    graph = AGraph(strict=False, directed=False)

    # Create nodes
    for k in range(nds.shape[0]):
        nd = np.where(nds[k]==1)[0]
        if len(nd) > 0:
            color = ID_COLOR[nd[0]+1]
            name = '' #CLASS_ROM[nd+1]
            graph.add_node(k, label=name, color=color)

    # Create edges
    for i, p, j in eds:
        if p > 0:
            graph.add_edge(i-shift, j-shift, color='black', penwidth='4')
    
    graph.node_attr['style']='filled'
    graph.layout(prog='dot')
    graph.draw('temp/_temp_{}.png'.format(EXP_ID))

    # Get array
    png_arr = open_png('temp/_temp_{}.png'.format(EXP_ID), im_size=im_size) 
    im_graph_tensor = torch.FloatTensor(png_arr.transpose(2, 0, 1)/255.0)
    return im_graph_tensor

def bb_to_img(bbs, graphs, room_to_sample, triple_to_sample, boundary_bb=None, max_num_nodes=10, im_size=512, disc_scores=None):
	imgs = []
	nodes, triples = graphs
	bbs = np.array(bbs)
	for k in range(bbs.shape[0]):
		
		# Draw graph image
		inds = torch.nonzero(triple_to_sample == k)[:, 0]
		tps = triples[inds]
		inds = torch.nonzero(room_to_sample == k)[:, 0]
		offset = torch.min(inds)
		nds = nodes[inds]
		
		s, p, o = tps.chunk(3, dim=1)          
		s, p, o = [x.squeeze(1) for x in [s, p, o]] 
		eds = torch.stack([s, o], dim=1)          
		eds = eds-offset

		# Draw BB image
		bb = bbs[k]
		nds = nodes.view(-1, max_num_nodes)[k, :].detach().cpu().numpy()
		image_arr = np.zeros((im_size, im_size))
		im = Image.fromarray(image_arr.astype('uint8')).convert('RGB')
		dr = ImageDraw.Draw(im)

		for box, node in zip(bb, nds):
			node = node+1
			x0, y0, x1, y1 = box
			color = ID_COLOR[node]
			dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline=color)

		image_tensor = torch.tensor(np.array(im).transpose(2, 0, 1)/255.0)
		imgs.append(image_tensor)

	imgs_tensor = torch.stack(imgs)

	return imgs_tensor

def mask_to_bb(mask):
    
    # get masks pixels
    inds = np.array(np.where(mask>0))
    
    if inds.shape[-1] == 0:
        return [0, 0, 0, 0]

    # Compute BBs
    y0, x0 = np.min(inds, -1)
    y1, x1 = np.max(inds, -1)

    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, 255), min(x1, 255)

    w = x1 - x0
    h = y1 - y0
    x, y = x0, y0
    
    return [x0, y0, x1+1, y1+1]

def extract_corners(bb1, bb2, im_size=256):

	# initialize
	corners_set = set()
	x0, y0, x1, y1 = bb1
	x2, y2, x3, y3 = bb2

	# add corners from bbs
	corners_set.add((int(x0*im_size), int(y0*im_size)))
	corners_set.add((int(x0*im_size), int(y1*im_size)))
	corners_set.add((int(x1*im_size), int(y0*im_size)))
	corners_set.add((int(x1*im_size), int(y1*im_size)))
	corners_set.add((int(x2*im_size), int(y2*im_size)))
	corners_set.add((int(x2*im_size), int(y3*im_size)))
	corners_set.add((int(x3*im_size), int(y2*im_size)))
	corners_set.add((int(x3*im_size), int(y3*im_size)))

	# add intersection corners
	es1 = [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]
	es2 = [(x2, y2, x3, y2), (x3, y2, x3, y3), (x3, y3, x2, y3), (x2, y3, x2, y2)]

	for e1 in es1:
		for e2 in es2:
			x0, y0, x1, y1 = e1
			x2, y2, x3, y3 = e2

			e1_im = Image.new('L', (im_size, im_size))
			dr = ImageDraw.Draw(e1_im)
			dr.line((x0*im_size, y0*im_size, x1*im_size, y1*im_size), fill='white', width=1)
			e1_im = np.array(e1_im)/255.0

			e2_im = Image.new('L', (im_size, im_size))
			dr = ImageDraw.Draw(e2_im)
			dr.line((x2*im_size, y2*im_size, x3*im_size, y3*im_size), fill='white', width=1)
			e2_im = np.array(e2_im)/255.0

			cs_inter = np.array(np.where(e1_im + e2_im > 1))
			if(cs_inter.shape[1] == 1):
				corners_set.add((cs_inter[1][0], cs_inter[0][0]))

	return corners_set


def align_bb(bbs_batch, th=0.03):
	new_bbs_batch = bbs_batch.copy()
# 	np.save('debug.npy', new_bbs_batch)
# 	new_bbs_batch = np.load('debug.npy')
	for bbs in new_bbs_batch:
		## DEBUG
# 		im_deb1 = Image.new('RGB', (256, 256))
# 		dr = ImageDraw.Draw(im_deb1)
# 		for i, bb in enumerate(bbs):
# 			x0, y0, x1, y1 = bb * 255.0
# 			if i != 6:
# 				dr.rectangle((x0, y0, x1, y1), outline='green')
# 			else:
# 				dr.rectangle((x0, y0, x1, y1), outline='white')
		## DEBUG

		for i, bb1 in enumerate(bbs):
			x0, y0, x1, y1 = bb1
			x0_avg, y0_avg, x1_avg, y1_avg = [], [], [], []
			tracker = []
			for j, bb2 in enumerate(bbs):
				x2, y2, x3, y3 = bb2
				# horizontals
				if abs(x2-x0) <= th:
					x0_avg.append(x2) 
					tracker.append((j, 0, 0))
				if abs(x3-x0) <= th:
					x0_avg.append(x3)
					tracker.append((j, 2, 0))
				if abs(x2-x1) <= th:
					x1_avg.append(x2)
					tracker.append((j, 0, 2))
				if abs(x3-x1) <= th:
					x1_avg.append(x3)
					tracker.append((j, 2, 2))
				# verticals
				if abs(y2-y0) <= th:
					y0_avg.append(y2)
					tracker.append((j, 1, 1))
				if abs(y3-y0) <= th:
					y0_avg.append(y3)
					tracker.append((j, 3, 1))
				if abs(y2-y1) <= th:
					y1_avg.append(y2)
					tracker.append((j, 1, 3))
				if abs(y3-y1) <= th:
					y1_avg.append(y3)
					tracker.append((j, 3, 3))
			avg_vec = [np.mean(x0_avg), np.mean(y0_avg), np.mean(x1_avg), np.mean(y1_avg)]
			for l, val in enumerate(avg_vec):
				if not np.isnan(avg_vec[l]):
					bbs[i, l] = avg_vec[l]
			for k, l, m in tracker:
				if not np.isnan(avg_vec[m]):
					bbs[k, l] = avg_vec[m]

# 		## DEBUG
# 		im_deb2 = Image.new('RGB', (256, 256))
# 		dr = ImageDraw.Draw(im_deb2)
# 		for bb in bbs:
# 			x0, y0, x1, y1 = bb * 255.0
# 			dr.rectangle((x0, y0, x1, y1), outline='red')
# 		## DEBUG
# 		im_deb1.save('deb_1.jpg')
# 		im_deb2.save('deb_2.jpg')
	return new_bbs_batch

def remove_junctions(junctions, juncs_on, lines_on, delta=10.0):

    curr_juncs_on, curr_lines_on = list(juncs_on), list(lines_on)
    while True:
        new_lines_on, new_juncs_on = [], []
        is_mod = False
        for j1 in curr_juncs_on:
            adj_js, adj_as, ls = [], [], []
            for j2 in curr_juncs_on:
                if ((j1, j2) in curr_lines_on) or ((j2, j1) in curr_lines_on):
                    adj_js.append(j2)
                    pt1 = junctions[j1]
                    pt2 = junctions[j2]
                    adj_as.append(getAngle(pt1, pt2))
                    ls.append((j1, j2))

            if len(adj_js) > 2 or is_mod or len(adj_js) == 1:
                new_juncs_on.append(j1)
                new_lines_on += ls
            elif len(adj_js) == 2:
                diff = np.abs(180.0-np.abs(adj_as[0]-adj_as[1]))
                if diff >= delta:
                    new_juncs_on.append(j1)
                    new_lines_on += ls
                else:
                    new_lines_on.append((adj_js[0], adj_js[1]))
                    is_mod = True
        curr_juncs_on, curr_lines_on = list(new_juncs_on), list(new_lines_on)
        if not is_mod:
            break

    return curr_juncs_on, curr_lines_on

def bb_to_seg(bbs_batch, im_size=256):

	all_rooms_batch = []
	for bbs in bbs_batch:
		areas = np.array([(x1-x0)*(y1-y0) for x0, y0, x1, y1 in bbs])
		inds = np.argsort(areas)[::-1]
		bbs = bbs[inds]
		tag = 1
		rooms_im = np.zeros((256, 256))

		for (x0, y0, x1, y1) in bbs:
			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
				continue
			else:
				room_im = Image.new('L', (256, 256))
				dr = ImageDraw.Draw(room_im)
				dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline='white', fill='white')
				inds = np.array(np.where(np.array(room_im) > 0))
				rooms_im[inds[1, :], inds[0, :]] = tag
				tag += 1

		all_rooms = []
		for tag in range(1, bbs.shape[0]+1):
			room = np.zeros((256, 256))
			inds = np.array(np.where(rooms_im == tag))
			room[inds[0, :], inds[1, :]] = 1.0
			all_rooms.append(room)
		all_rooms_batch.append(all_rooms)
	all_rooms_batch = np.array(all_rooms_batch)

# 	edges_batch = []
# 	for b in range(all_rooms_batch.shape[0]):
# 		edge_arr = []
# 		for k in range(all_rooms_batch.shape[1]):
# 			rm_arr = all_rooms_batch[b, k, :, :]
# 			rm_im = Image.fromarray(rm_arr*255)
# 			rm_im_lg = rm_im.filter(ImageFilter.MaxFilter(5))
# 			rm_im_sm = rm_im.filter(ImageFilter.MinFilter(5))
# 			edge_arr.append(np.array(rm_im_lg) - np.array(rm_im_sm))
# 		edges_batch.append(edge_arr)
# 	edges_batch = np.array(edges_batch)
# 	print(edges_batch.shape)

	return all_rooms_batch

def bb_to_im_fid(bbs_batch, nodes, im_size=299):
  nodes = np.array(nodes)
  bbs = np.array(bbs_batch[0])
  areas = np.array([(x1-x0)*(y1-y0) for x0, y0, x1, y1 in bbs])
  inds = np.argsort(areas)[::-1]
  bbs = bbs[inds]
  nodes = nodes[inds]
  im = Image.new('RGB', (im_size, im_size), 'white')
  dr = ImageDraw.Draw(im)
  for (x0, y0, x1, y1), nd in zip(bbs, nodes):
      if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
          continue
      else:
          color = ID_COLOR[int(nd)+1]
          dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), width=3, outline='black', fill=color)
  return im

# def bb_to_seg(bbs_batch, im_size=256, num_bbs=10):

# 	all_rooms_batch = []
# 	for bbs in bbs_batch:
# 		bbs = bbs.reshape(num_bbs, 4)
# 		inds = list(range(num_bbs))
# 		random.shuffle(inds)
# 		bbs = bbs[inds]
# 		tag = 1
# 		rooms_im = np.zeros((256, 256))

# 		for (x0, y0, x1, y1) in bbs:
# 			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
# 				continue
# 			else:
# 				room_im = Image.new('L', (256, 256))
# 				dr = ImageDraw.Draw(room_im)
# 				dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline='white', fill='white')
# 				inds = np.array(np.where(np.array(room_im) > 0))
# 				rooms_im[inds[0, :], inds[1, :]] = tag
# 				tag += 1

# 		all_rooms = []
# 		for tag in range(1, bbs.shape[0]+1):
# 			room = np.zeros((256, 256))
# 			inds = np.array(np.where(rooms_im == tag))
# 			room[inds[0, :], inds[1, :]] = 1.0
# 			all_rooms.append(room)
# 		all_rooms_batch.append(all_rooms)
# 	all_rooms_batch = np.array(all_rooms_batch)

# 	all_rooms_sm_batch = []
# 	edges_batch = []
# 	for b in range(all_rooms_batch.shape[0]):
# 		edge_arr = np.zeros((256, 256))
# 		all_rooms_sm = []
# 		for k in range(all_rooms_batch.shape[1]):
# 			rm_arr = all_rooms_batch[b, k, :, :]
# 			rm_im = Image.fromarray(rm_arr*255)
# 			rm_im_lg = rm_im.filter(ImageFilter.MaxFilter(5))
# 			rm_im_sm = rm_im.filter(ImageFilter.MinFilter(5))
# 			all_rooms_sm.append(np.array(rm_im_sm)/255.0)
# 			edge_arr += np.array(rm_im_lg) - np.array(rm_im_sm)
# 		edge_arr = np.clip(edge_arr, 0, 255)
# 		edges_batch.append(edge_arr/255.0)
# 		all_rooms_sm_batch.append(all_rooms_sm)
# 	edges_batch = np.array(edges_batch)[:, np.newaxis, :, :]

# 	all_rooms_sm_batch = np.array(all_rooms_sm_batch)
# 	all_rooms_sm_batch = np.sum(all_rooms_sm_batch, 1)[:, np.newaxis, :, :]
# 	all_rooms_sm_batch = np.clip(all_rooms_sm_batch, 0, 1)
# 	edges_batch = np.concatenate([np.zeros((all_rooms_sm_batch.shape[0], 3, 256, 256)), all_rooms_sm_batch, np.zeros((all_rooms_sm_batch.shape[0], 7, 256, 256)), edges_batch], 1)

# # 	edges_batch = np.concatenate([np.zeros((all_rooms_batch.shape[0], 1, 256, 256)), edges_batch], 1)
# # 	all_rooms_batch = np.concatenate([all_rooms_batch, edges_batch], 1)

# 	return edges_batch

def get_type(pxs):
	ori_arr = [0, 0, 0, 0]
	for p in pxs:
		if tuple(p) == (0, 1):
			ori_arr[0] = 1
		if tuple(p) == (1, 2):
			ori_arr[1] = 1
		if tuple(p) == (2, 1):
			ori_arr[2] = 1
		if tuple(p) == (1, 0):
			ori_arr[3] = 1

# POINT_ORIENTATIONS = [[(2, ), (3, ), (0, ), (1, )], [(0, 3), (0, 1), (1, 2), (2, 3)], [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)], [(0, 1, 2, 3)]]
	# Type 1
	if tuple(ori_arr) == tuple([0, 0, 1, 0]):
		return 0
	# Type 2
	if tuple(ori_arr) == tuple([0, 0, 0, 1]):
		return 1
	# Type 3
	if tuple(ori_arr) == tuple([1, 0, 0, 0]):
		return 2
	# Type 4
	if tuple(ori_arr) == tuple([0, 1, 0, 0]):
		return 3
	# Type 5
	if tuple(ori_arr) == tuple([1, 0, 0, 1]):
		return 4
	# Type 6
	if tuple(ori_arr) == tuple([1, 1, 0, 0]):
		return 5
	# Type 7
	if tuple(ori_arr) == tuple([0, 1, 1, 0]):
		return 6
	# Type 8
	if tuple(ori_arr) == tuple([0, 0, 1, 1]):
		return 7
	# Type 9
	if tuple(ori_arr) == tuple([0, 1, 1, 1]):
		return 8
	# Type 10
	if tuple(ori_arr) == tuple([1, 0, 1, 1]):
		return 9
	# Type 11
	if tuple(ori_arr) == tuple([1, 1, 0, 1]):
		return 10
	# Type 12
	if tuple(ori_arr) == tuple([1, 1, 1, 0]):
		return 11
	# Type 13
	if tuple(ori_arr) == tuple([1, 1, 1, 1]):
		return 12

def bb_to_vec(bbs_batch, im_size=256):
	cs_type_batch = []
	for bbs in bbs_batch:
		corners_set = set()
		for x0, y0, x1, y1 in bbs:
			x0, y0, x1, y1 = int(x0*255.0), int(y0*255.0), int(x1*255.0), int(y1*255.0)
			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
				continue
			else:
				corners_set.add((x0, y0))
				corners_set.add((x0, y1))
				corners_set.add((x1, y0))
				corners_set.add((x1, y1))
# 		corners_set_aug = set()
# 		for x0, y0 in list(corners_set):
# 			for x1, y1 in list(corners_set):
# 				corners_set_aug.add((x0, y0))
# 				corners_set_aug.add((x0, y1))
# 				corners_set_aug.add((x1, y0))
# 				corners_set_aug.add((x1, y1))
		cs_type_batch.append(list(corners_set))
	return cs_type_batch

# def bb_to_vec(bbs_batch, im_size=256, num_bbs=10):
# 	bbs_batch = bbs_batch.detach().cpu().numpy()
# 	cs_type_batch = []
# 	cs_batch = []
# 	for bbs in bbs_batch:
# 		bbs = bbs.reshape(num_bbs, 4)
# 		corners_set = set()
# 		for (x0, y0, x1, y1) in bbs:
# 			for (x2, y2, x3, y3) in bbs:
# 				bb1 = (x0, y0, x1, y1)
# 				bb2 = (x2, y2, x3, y3)
# 				if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
# 					continue
# 				elif x2 < 0 or y2 < 0 or x3 < 0 or y3 < 0:
# 					continue
# 				else:
# 					corners_set = corners_set.union(extract_corners(bb1, bb2))

# 		bbs_im = Image.new('L', (256, 256))
# 		dr = ImageDraw.Draw(bbs_im)
# 		for (x0, y0, x1, y1) in bbs:
# 			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
# 				continue
# 			else:
# 				dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline='white')

# # 		bbs_im.save('./debug0.jpg')
# 		bbs_im = np.array(bbs_im)
# 		corners_set = np.array(list(corners_set))
# 		cs_type_sample = [[], [], [], [], [], [], [], [], [], [], [], [], []]
# 		for c in corners_set:
# 			y, x = c
# 			c_im = np.zeros((256, 256))
# 			c_im[x, y] = 255
# 			c_im = Image.fromarray(c_im.astype('uint8'))
# # 			print(x-1, x+2, y-1, y+2)
# 			pxs = np.array(np.where(bbs_im[x-1:x+2, y-1:y+2] > 0)).transpose()
# 			if(pxs.shape[0] == 0):
# 				print(bbs_im[x-1:x+2, y-1:y+2])
# 				print(x, y)
				
# 			_type = get_type(pxs)
# 			if _type is not None:
# 				cs_type_sample[_type].append((x, y))
# 		cs_type_batch.append(cs_type_sample)

# # 		# debug
# # 		bbs_im_debug = Image.new('L', (256, 256))
# # 		dr = ImageDraw.Draw(bbs_im_debug)
# # 		for (x0, y0, x1, y1) in bbs:
# # 			if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
# # 				continue
# # 			else:
# # 				dr.rectangle((x0*im_size, y0*im_size, x1*im_size, y1*im_size), outline='white')
# # 		bbs_im_debug.save('./debug1.jpg')

# # 		corners_d ebug = Image.new('L', (256, 256))
# # 		dr = ImageDraw.Draw(corners_debug)
# # 		for x, y in list(corners_set):
# # 			dr.ellipse((x-2, y-2, x+2, y+2), outline='white')
# # 		corners_debug.save('./debug2.jpg')
# # 		print(corners_set)
# # 		print(np.array(list(corners_set)).shape)

# 	return cs_type_batch

def  visualizeCorners(wallPoints):
	im_deb = Image.new('RGB', (256, 256))
	dr = ImageDraw.Draw(im_deb)
	for (x, y, i, j) in wallPoints:
		dr.ellipse((x-1, y-1, x+1, y+1), fill='red')
		font = ImageFont.truetype("arial.ttf", 10)
		dr.text((x, y), str(3*i+j+1),(255,255,255), font=font)
	im_deb.save('./debug_all_corner_with_text.jpg')
	return

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
def rectangle_renderer(theta, im_size=64):
    
    # scale theta
    theta = theta*im_size
    
    # create meshgrid
    xs = np.arange(im_size)
    ys = np.arange(im_size)
    xs, ys = np.meshgrid(xs, ys)
    xs = torch.tensor(np.repeat(xs[np.newaxis, :, :], theta.shape[0], axis=0)).float().cuda()
    ys = torch.tensor(np.repeat(ys[np.newaxis, :, :], theta.shape[0], axis=0)).float().cuda()

    # conditions
    cond_1 = torch.min(torch.cat([F.relu(ys - theta[:, 1].view(-1, 1, 1)).unsqueeze(-1), torch.ones((theta.shape[0], im_size, im_size, 1)).cuda()], -1), -1)[0] * \
             torch.min(torch.cat([F.relu(theta[:, 3].view(-1, 1, 1) - ys).unsqueeze(-1), torch.ones((theta.shape[0], im_size, im_size, 1)).cuda()], -1), -1)[0]
            
    cond_2 = torch.min(torch.cat([F.relu(xs - theta[:, 0].view(-1, 1, 1)).unsqueeze(-1), torch.ones((theta.shape[0], im_size, im_size, 1)).cuda()], -1), -1)[0] * \
             torch.min(torch.cat([F.relu(theta[:, 2].view(-1, 1, 1) - xs).unsqueeze(-1), torch.ones((theta.shape[0], im_size, im_size, 1)).cuda()], -1), -1)[0]

    # lines
    line_1 = (F.relu(torch.ones((theta.shape[0], im_size, im_size)).cuda() - torch.abs(xs - torch.ones((theta.shape[0], im_size, im_size)).cuda() - theta[:, 0].view(-1, 1, 1))) * cond_1).view(-1, im_size, im_size, 1)    # top
    line_2 = (F.relu(torch.ones((theta.shape[0], im_size, im_size)).cuda() - torch.abs(xs + torch.ones((theta.shape[0], im_size, im_size)).cuda() - theta[:, 2].view(-1, 1, 1))) * cond_1).view(-1, im_size, im_size, 1)    # bottom
    line_3 = (F.relu(torch.ones((theta.shape[0], im_size, im_size)).cuda() - torch.abs(ys - theta[:, 1].view(-1, 1, 1))) * cond_2).view(-1, im_size, im_size, 1)        # left
    line_4 = (F.relu(torch.ones((theta.shape[0], im_size, im_size)).cuda() - torch.abs(ys - theta[:, 3].view(-1, 1, 1))) * cond_2).view(-1, im_size, im_size, 1)        # right
            
    I = torch.max(torch.cat([line_1, line_2, line_3, line_4], -1), -1)[0]
    
    return I

def checkpoint(real_room_bb, fake_room_bb,  nodes, triples, room_to_sample, triple_to_sample, generator, exp_folder, batches_done, fake_validity, real_validity, boundary_bb, Tensor, latent_dim, out_imsize):
    
    torch.save(generator.state_dict(), './checkpoints/gen_neighbour_{}_{}.pth'.format(exp_folder, batches_done))
    fake_imgs_tensor = bb_to_img(fake_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample, \
                                 boundary_bb, disc_scores=fake_validity, im_size=out_imsize)
    real_imgs_tensor = bb_to_img(real_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample, \
                                 boundary_bb, disc_scores=real_validity, im_size=out_imsize)


    save_image(fake_imgs_tensor, "{}/fake_{}.png".format(exp_folder, batches_done), nrow=16)
    save_image(real_imgs_tensor, "{}/real_{}.png".format(exp_folder, batches_done), nrow=16)

    ## perform variation analysis
    # Sample noise as generator input
    layouts_imgs_tensor = []
    n_samples = 16
    for _ in range(10):

        # get partial batch
        z = Variable(Tensor(np.random.normal(0, 1, (real_room_bb.shape[0], latent_dim))))
        z_partial = z[:n_samples]
        nodes_partial = nodes[:n_samples*10]
        triples_partial = triples[:n_samples*45, :]
        room_to_sample_partial = room_to_sample[:n_samples*10]
        boundary_bb_partial = boundary_bb[:n_samples, :]
        triple_to_sample_partial = triple_to_sample[:n_samples*45]

        # plot images
        fake_room_bb_partial = generator(z_partial, [nodes_partial, triples_partial], room_to_sample_partial, boundary=boundary_bb_partial)
        fake_imgs_tensor = bb_to_img(fake_room_bb_partial.data, [nodes_partial, triples_partial], room_to_sample_partial, \
                                     triple_to_sample_partial, boundary_bb_partial, im_size=out_imsize)

        layouts_imgs_tensor.append(fake_imgs_tensor)
    layouts_imgs_tensor = torch.stack(layouts_imgs_tensor)
    layouts_imgs_tensor = layouts_imgs_tensor.view(10, 16, 2, 3, out_imsize, out_imsize)
    layouts_imgs_tensor_filtered = []
    for k in range(16):
        for l in range(10):
            if l == 0:
                layouts_imgs_tensor_filtered.append(layouts_imgs_tensor[l, k, 0, :, :, :])
            layouts_imgs_tensor_filtered.append(layouts_imgs_tensor[l, k, 1, :, :, :])
    layouts_imgs_tensor_filtered = torch.stack(layouts_imgs_tensor_filtered).contiguous().view(-1, 3, out_imsize, out_imsize)
    save_image(layouts_imgs_tensor_filtered, "{}/layouts_{}.png".format(exp_folder, batches_done), nrow=11)

# def combine_images(layout_batch, im_size=256):
#     layout_batch = layout_batch.detach().cpu().numpy()
#     all_imgs = []
#     for layout in layout_batch:
#         comb_img = Image.new('RGB', (im_size, im_size))
#         dr = ImageDraw.Draw(comb_img)
#         for i in range(layout.shape[1]):
#             for j in range(layout.shape[2]):
#                 h, w = layout[0, i, j], layout[1, i, j]
#                 if layout[2, i, j] > 0.5:
#                     label = 1 #np.argmax(layout[:10, i, j]) + 1
#                     h, w = layout[0, i, j], layout[1, i, j]
#                     color = ID_COLOR[int(label)]
#                     r = im_size/layout.shape[1]
#                     dr.rectangle((r*i-(im_size*h)/2.0, r*j-(im_size*w)/2.0, \
#                                   r*i+(im_size*h)/2.0, r*j+(im_size*w)/2.0), outline=color)
#         all_imgs.append(torch.tensor(np.array(comb_img).\
#                                      astype('float').\
#                                      transpose(2, 0, 1))/255.0)
#     all_imgs = torch.stack(all_imgs)
#     return all_imgs
            

def combine_images_bbs(bbs_batch, im_size=256):
    bbs_batch = bbs_batch.view(-1, 10, 4).detach().cpu().numpy()
    all_imgs = []
    for bbs in bbs_batch:
        comb_img = Image.new('RGB', (im_size, im_size))
        dr = ImageDraw.Draw(comb_img)
        for bb in bbs:
            x0, y0, x1, y1 = im_size*bb
            h = x1-x0
            w = y1-y0
            if h > 4 and w > 4:
                color = ID_COLOR[1]
                dr.rectangle((x0, y0, x1, y1), outline=color)
        all_imgs.append(torch.tensor(np.array(comb_img).\
                                     astype('float').\
                                     transpose(2, 0, 1))/255.0)
    all_imgs = torch.stack(all_imgs)
    return all_imgs

import webcolors
def combine_images_maps(maps_batch, nodes_batch, edges_batch, \
                        nd_to_sample, ed_to_sample, im_size=256):
    maps_batch = maps_batch.detach().cpu().numpy()
    nodes_batch = nodes_batch.detach().cpu().numpy()
    edges_batch = edges_batch.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    
    all_imgs = []
    shift = 0
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b)
        inds_ed = np.where(ed_to_sample==b)
        
        mks = maps_batch[inds_nd]
        nds = nodes_batch[inds_nd]
        eds = edges_batch[inds_ed]
        
        comb_img = np.ones((im_size, im_size, 3)) * 255
        extracted_rooms = []
        for mk, nd in zip(mks, nds):
            r =  im_size/mk.shape[-1]
            x0, y0, x1, y1 = np.array(mask_to_bb(mk)) * r 
            h = x1-x0
            w = y1-y0
            if h > 0 and w > 0:
                extracted_rooms.append([mk, (x0, y0, x1, y1), nd])
        
        # draw graph
        graph_img = draw_graph(nds, eds, shift, im_size=im_size)
        shift += len(nds)
        all_imgs.append(graph_img)
        
        # draw masks
        mask_img = np.ones((32, 32, 3)) * 255
        for rm in extracted_rooms:
            mk, _, nd = rm 
            inds = np.array(np.where(mk>0))
            _type = np.where(nd==1)[0]
            if len(_type) > 0:
                color = ID_COLOR[_type[0] + 1]
            else:
                color = 'black'
            r, g, b = webcolors.name_to_rgb(color)
            mask_img[inds[0, :], inds[1, :], :] = [r, g, b]
        mask_img = Image.fromarray(mask_img.astype('uint8'))
        mask_img = mask_img.resize((im_size, im_size))
        all_imgs.append(torch.FloatTensor(np.array(mask_img).transpose(2, 0, 1))/255.0)
            
        # draw boxes - filling
        comb_img = Image.fromarray(comb_img.astype('uint8'))
        dr = ImageDraw.Draw(comb_img)
        for rm in extracted_rooms:
            _, rec, nd = rm 
            dr.rectangle(tuple(rec), fill='beige')
            
        # draw boxes - outline
        for rm in extracted_rooms:
            _, rec, nd = rm 
            _type = np.where(nd==1)[0]
            if len(_type) > 0:
                color = ID_COLOR[_type[0] + 1]
            else:
                color = 'black'
            dr.rectangle(tuple(rec), outline=color, width=4)
            
#         comb_img = comb_img.resize((im_size, im_size))
        all_imgs.append(torch.FloatTensor(np.array(comb_img).\
                                     astype('float').\
                                     transpose(2, 0, 1))/255.0)
    all_imgs = torch.stack(all_imgs)
    return all_imgs
