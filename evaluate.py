import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from floorplan_dataset_no_masks import FloorplanGraphDataset, floorplan_collate_fn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from graph import GraphTripleConv, GraphTripleConvNet

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from utils import bb_to_img, bb_to_vec, bb_to_seg
from PIL import Image, ImageDraw
from MyIP import reconstructFloorplan
import svgwrite

from models import Generator
import networkx as nx
import matplotlib.pyplot as plt
from utils import ID_COLOR
from floorplan_dataset_no_masks import is_adjacent
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=20, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--img_size", type=int, default=4, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--with_boundary", action='store_true', default=True, help="include floorplan footprint")
parser.add_argument("--num_variations", type=int, default=10, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")
# parser.add_argument("--checkpoint", type=str, default='checkpoints/gen_neighbour_exp_10_nodes_train_split_1000000.pth', help="destination folder")


opt = parser.parse_args()
print(opt)

def return_eq(node1, node2):
    return node1['label']==node2['label']

def compute_dist(bb1, bb2):

	x0, y0, x1, y1 = bb1
	x2, y2, x3, y3 = bb2 

	h1, h2 = x1-x0, x3-x2
	w1, w2 = y1-y0, y3-y2

	xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
	yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0

	delta_x = abs(xc2-xc1) - (h1 + h2)/2.0
	delta_y = abs(yc2-yc1) - (w1 + w2)/2.0

	return delta_x, delta_y


def retrieve_connections(nodes, gen_room_bb):
	edges = []
	nodes = [x for x in nodes.detach().cpu().numpy() if x >= 0]
	gen_room_bb = gen_room_bb.view(-1, 4).detach().cpu().numpy()
	for k, bb1 in enumerate(gen_room_bb):
		x0, y0, x1, y1 = bb1 * 256.0
		for l, bb2 in enumerate(gen_room_bb):
			x2, y2, x3, y3 = bb2 * 256.0
			if (x0 >= 0) and (y0 >= 0) and (x1 >= 0) and (y1 >= 0):
				if (x2 >= 0) and (y2 >= 0) and (x3 >= 0) and (y3 >= 0):
					if k > l:
						if is_adjacent(bb1, bb2):
							edges.append((k, l))
	return nodes, edges

def draw_floorplan(dwg, junctions, juncs_on, lines_on):

    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])/2.0
        x2, y2 = np.array(junctions[l])/2.0
        #fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=4, opacity=0.5))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j])/2.0
        dwg.add(dwg.circle(center=(x, y), r=2, stroke='red', fill='white', stroke_width=1, opacity=0.75))
    return 

# Initialize generator and discriminator
generator = Generator(opt.with_boundary)
generator.load_state_dict(torch.load(opt.checkpoint))
generator.eval()

# Initialize variables
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = '/local-scratch2/nnauata/autodesk/FloorplanDataset/'

# Initialize dataset iterator
fp_dataset = FloorplanGraphDataset(rooms_path, split='test')
fp_loader = torch.utils.data.DataLoader(fp_dataset, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False, collate_fn=floorplan_collate_fn)
fp_iter = tqdm(fp_loader, total=len(fp_dataset) // opt.batch_size + 1)

# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------------
#  Vectorize
# ------------
edit_dist_per_node = defaultdict(list)
globalIndex = 0
for i, batch in enumerate(fp_iter):

	# Unpack batch
	room_bb, boundary_bb, nodes, triples, room_to_sample, _ = batch

	# Configure input
	real_room_bb = Variable(room_bb.type(Tensor))
	boundary_bb = Variable(boundary_bb.type(Tensor))

	# Sample noise as generator input
	layouts_imgs_tensor = []

	# plot images
# 	np.random.seed(100)
	z = Variable(Tensor(np.random.normal(0, 1, (real_room_bb.shape[0], opt.latent_dim))))
	gen_room_bb = generator(z, [nodes, triples], room_to_sample, boundary=boundary_bb)
	nodes = nodes.view(real_room_bb.shape[0], -1)
	triples = triples.view(real_room_bb.shape[0], -1, 3)
	real_room_bb = real_room_bb.detach().cpu().numpy()

	# reconstruct
	for i in range(gen_room_bb.shape[0]):

		g_pred = retrieve_connections(nodes[i], gen_room_bb[i])
		g_true = [nodes[i].detach().cpu().numpy(), triples[i].detach().cpu().numpy()]

		G = nx.Graph()
		colors_G = []
		for k, label in enumerate(g_pred[0]):
			G.add_nodes_from([(k, {'label':label})])
			colors_G.append(ID_COLOR[label])
		for k, l in g_pred[1]:
			G.add_edges_from([(k, l)])

		H = nx.Graph()
		colors_H = []
		for k, label in enumerate(g_true[0]):
			if label >= 0:
				H.add_nodes_from([(k, {'label':label})])
				colors_H.append(ID_COLOR[label])

		for k, m, l in g_true[1]:
			if m >= 0:
# 				print((k, l))
				H.add_edges_from([(k, l)])
    
		min_dist =np.min([x for x in nx.optimize_graph_edit_distance(G, H)])
		edit_dist_per_node[len(H.nodes())].append(min_dist)
		globalIndex += 1
        
# 		# save predictions
# 		im_pred = Image.new('RGB', (256, 256))
# 		dr = ImageDraw.Draw(im_pred)
# 		bbs = gen_room_bb[i].view(-1, 4)
# 		for nd, bb in zip(nodes[i].detach().cpu().numpy(), bbs):
# 			x0, y0, x1, y1 = bb * 256.0
# 			if x0 >= 0 and y0 >= 0 and x1 >= 0 and y1 >= 0:
# 				color = ID_COLOR[nd]
# 				dr.rectangle((x0, y0, x1, y1), outline=color)
# 		im_pred.save('./debug/{}_pred_bb.jpg'.format(globalIndex))

# 		# save predictions
# 		im_true = Image.new('RGB', (256, 256))
# 		dr = ImageDraw.Draw(im_true)
# 		bbs = real_room_bb[i].reshape((-1, 4))
# 		for nd, bb in zip(nodes[i].detach().cpu().numpy(), bbs):
# 			x0, y0, x1, y1 = bb * 256.0
# 			if x0 >= 0 and y0 >= 0 and x1 >= 0 and y1 >= 0:
# 				color = ID_COLOR[nd]
# 				dr.rectangle((x0, y0, x1, y1), outline=color)
# 		im_true.save('./debug/{}_true_bb.jpg'.format(globalIndex))

# 		plt.figure()
# 		pos = nx.spring_layout(G)
# 		nx.draw(G, pos, node_size=1000, node_color=colors_G, font_size=8, font_weight='bold')
# 		plt.tight_layout()
# 		plt.show()
# 		plt.savefig('./debug/{}_pred_graph.jpg'.format(globalIndex), format="PNG")

# 		plt.figure()
# 		pos = nx.spring_layout(H)
# 		colors = [ID_COLOR[nd] for nd in H]
# 		nx.draw(H, pos, node_size=1000, node_color=colors_H, font_size=8, font_weight='bold')
# 		plt.tight_layout()
# 		plt.show()
# 		plt.savefig('./debug/{}_true_graph.jpg'.format(globalIndex), format="PNG")


for n in edit_dist_per_node:
	print(n, len(edit_dist_per_node[n]), np.mean(edit_dist_per_node[n]))


# import networkx as nx

# G=nx.Graph()
# G.add_nodes_from([("A", {'label':'a'}), ("B", {'label':'b'}),
#                   ("C", {'label':'c'})])

# G.add_edges_from([("A","B"),("A","C")])

# H=nx.Graph()
# H.add_nodes_from([("X", {'label':'x'}), ("Y", {'label':'y'}),
#                   ("Z", {'label':'z'})])
# H.add_edges_from([("X","Y"),("X","Z")])

# # This is the function which checks for equality of labels
# def return_eq(node1, node2):
#     return node1['label']==node2['label']

# print(nx.graph_edit_distance(G, H, node_match=return_eq))
# # Output: 3