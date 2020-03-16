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
import math
import numpy as np
import PIL
from skimage.transform import resize as imresize
import pycocotools.mask as mask_utils
import glob
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt
import random
from utils import mask_to_bb, ROOM_CLASS 
sets = {'A':[1, 3], 'B':[4, 6], 'C':[7, 9], 'D':[10, 12], 'E':[13, 100]}

def filter_graphs(graphs, min_h=0.03, min_w=0.03):
    new_graphs = []
    for g in graphs:
        
        # retrieve data
        rooms_type = g[0]
        rooms_bbs = g[1]
        
        # discard broken samples
        check_none = np.sum([bb is None for bb in rooms_bbs])
        check_node = np.sum([nd == 0 for nd in rooms_type])
        if (len(rooms_type) == 0) or (check_none > 0) or (check_node > 0):
            continue
        
        # filter small rooms
        tps_filtered = []
        bbs_filtered = []
        for n, bb in zip(rooms_type, rooms_bbs):
            h, w = (bb[2]-bb[0]), (bb[3]-bb[1])
            if h > min_h and w > min_w:
                tps_filtered.append(n)
                bbs_filtered.append(bb)
        
        # update graph
        g_new = [tps_filtered, bbs_filtered]
        new_graphs.append(g_new)
    return new_graphs

class FloorplanGraphDataset(Dataset):
	def __init__(self, shapes_path, transform=None, target_set=None, split='train'):
		super(Dataset, self).__init__()
		self.shapes_path = shapes_path
		self.split = split
		self.target_set = target_set
		if split == 'train':
			self.subgraphs = np.load('{}/train_data.npy'.format(self.shapes_path), allow_pickle=True)
			self.augment = True
		elif split == 'eval':
			self.subgraphs = np.load('{}/train_data.npy'.format(self.shapes_path), allow_pickle=True)
			self.augment = False
		else:
			print('Error split not supported')        
			exit(1)
		self.transform = transform
		self.subgraphs = filter_graphs(self.subgraphs)
        
		# filter samples
		min_N = sets[self.target_set][0]
		max_N = sets[self.target_set][1]
		filtered_subgraphs = []
		for g in self.subgraphs:
			rooms_type = g[0]    
			in_set = (len(rooms_type) >= min_N) and (len(rooms_type) <= max_N)
			if (split == 'train') and (in_set == False):
				filtered_subgraphs.append(g)
			elif (split == 'eval') and (in_set == True):
				filtered_subgraphs.append(g)
		self.subgraphs = filtered_subgraphs
		if split == 'eval':
			self.subgraphs = self.subgraphs[:5000] # max 5k
		print(len(self.subgraphs))   
        
		# doblecheck
		deb_dic = defaultdict(int)
		for g in self.subgraphs:
			rooms_type = g[0] 
			if len(rooms_type) > 0:
				deb_dic[len(rooms_type)] += 1
		print("target samples:", deb_dic)
        
	def __len__(self):
		return len(self.subgraphs)

	def __getitem__(self, index):

		# load data
		graph = self.subgraphs[index]
		rooms_type = graph[0]
		rooms_bbs = graph[1]

		if self.augment:
			rot = random.randint(0, 3)*90.0
			flip = random.randint(0, 1) == 1
			rooms_bbs_aug = []
			for bb in rooms_bbs:
				x0, y0 = self.flip_and_rotate(np.array([bb[0], bb[1]]), flip, rot)
				x1, y1 = self.flip_and_rotate(np.array([bb[2], bb[3]]), flip, rot)
				xmin, ymin = min(x0, x1), min(y0, y1)
				xmax, ymax = max(x0, x1), max(y0, y1)
				rooms_bbs_aug.append(np.array([xmin, ymin, xmax, ymax]).astype('int'))
			rooms_bbs = rooms_bbs_aug
		rooms_bbs = np.stack(rooms_bbs)

# 		# make orderFloorplanDataset
# 		order_inds = [x[0] for x in sorted(enumerate(rooms_bbs), key=lambda bb:bb[1][1] + bb[1][0] * 256)]
# 		rooms_bbs = rooms_bbs[order_inds]/256.0
# 		rooms_type = [rooms_type[i] for i in order_inds]
		rooms_bbs = rooms_bbs/256.0

		# extract boundary box and centralize
		tl = np.min(rooms_bbs[:, :2], 0)
		br = np.max(rooms_bbs[:, 2:], 0)
		shift = (tl+br)/2.0 - 0.5
		rooms_bbs[:, :2] -= shift
		rooms_bbs[:, 2:] -= shift
		tl -= shift
		br -= shift
		boundary_bb = np.concatenate([tl, br])

		# build input graph
		rooms_bbs, nodes, edges = self.build_graph(rooms_bbs, rooms_type) 
		im_size = 32
		rooms_mks = np.zeros((nodes.shape[0], im_size, im_size))
		for k, (rm, bb) in enumerate(zip(nodes, rooms_bbs)):
			if rm > 0:
				x0, y0, x1, y1 = im_size*bb
				x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
				rooms_mks[k, x0:x1+1, y0:y1+1] = 1.0
		
		nodes = one_hot_embedding(nodes)[:, 1:]
		nodes = torch.FloatTensor(nodes)
		edges = torch.LongTensor(edges)
		rooms_mks = torch.FloatTensor(rooms_mks)
		rooms_mks = self.transform(rooms_mks)
		return rooms_mks, nodes, edges
        
	def flip_and_rotate(self, v, flip, rot, shape=256.):
		v = self.rotate(np.array((shape, shape)), v, rot)
		if flip:
			x, y = v
			v = (shape/2-abs(shape/2-x), y) if x > shape/2 else (shape/2+abs(shape/2-x), y)
		return v
	
	# rotate coords
	def rotate(self, image_shape, xy, angle):
		org_center = (image_shape-1)/2.
		rot_center = (image_shape-1)/2.
		org = xy-org_center
		a = np.deg2rad(angle)
		new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
				-org[0]*np.sin(a) + org[1]*np.cos(a) ])
		new = new+rot_center
		return new

	def build_graph(self, bbs, types):
		# create edges -- make order
		triples = []
		nodes = types
		bbs = np.array(bbs)
        
		# encode connections
		for k in range(len(nodes)):
			for l in range(len(nodes)):
				if l > k:
					nd0, bb0 = nodes[k], bbs[k]
					nd1, bb1 = nodes[l], bbs[l]
					if is_adjacent(bb0, bb1):
						if 'train' in self.split:
							triples.append([k, 1, l])
						else:
							triples.append([k, 1, l])
					else:
						if 'train' in self.split:
							triples.append([k, -1, l])
						else:
							triples.append([k, -1, l])

		# convert to array
		nodes = np.array(nodes)
		triples = np.array(triples)
		bbs = np.array(bbs)
		return bbs, nodes, triples

def _augment(mks):

	flip = random.choice([False, True])
	rot = random.choice([0, 90, 180, 270])
	new_mks = []
	for m in mks:
		m_im = Image.fromarray(m.astype('uint8'))
		m_im = m_im.rotate(rot)
		if flip:
			m_im = m_im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
		new_mks.append(np.array(m_im))
	new_mks = np.stack(new_mks)

	return new_mks

def is_adjacent(box_a, box_b, threshold=0.03):
	
	x0, y0, x1, y1 = box_a
	x2, y2, x3, y3 = box_b

	h1, h2 = x1-x0, x3-x2
	w1, w2 = y1-y0, y3-y2

	xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
	yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0

	delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
	delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0

	delta = max(delta_x, delta_y)

	return delta < threshold

def one_hot_embedding(labels, num_classes=11):
	"""Embedding labels to one-hot form.

	Args:
	  labels: (LongTensor) class labels, sized [N,].
	  num_classes: (int) number of classes.

	Returns:
	  (tensor) encoded labels, sized [N, #classes].
	"""
	y = torch.eye(num_classes) 
	return y[labels] 

def floorplan_collate_fn(batch):
	all_rooms_mks, all_nodes, all_edges = [], [], []
	all_node_to_sample, all_edge_to_sample = [], []
	node_offset = 0
	for i, (rooms_mks, nodes, edges) in enumerate(batch):
		O, T = nodes.size(0), edges.size(0)
		all_rooms_mks.append(rooms_mks)
		all_nodes.append(nodes)
		edges = edges.clone()
		if edges.shape[0] > 0:
			edges[:, 0] += node_offset
			edges[:, 2] += node_offset
			all_edges.append(edges)
		all_node_to_sample.append(torch.LongTensor(O).fill_(i))
		all_edge_to_sample.append(torch.LongTensor(T).fill_(i))
		node_offset += O
	all_rooms_mks = torch.cat(all_rooms_mks, 0)
	all_nodes = torch.cat(all_nodes)
	if len(all_edges) > 0:
		all_edges = torch.cat(all_edges)
	else:
		all_edges = torch.tensor([])       
	all_node_to_sample = torch.cat(all_node_to_sample)
	all_edge_to_sample = torch.cat(all_edge_to_sample)
	return all_rooms_mks, all_nodes, all_edges, all_node_to_sample, all_edge_to_sample

