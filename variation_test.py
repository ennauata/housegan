import argparse
import os
import numpy as np
import math
import sys
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image

from floorplan_dataset_maps import FloorplanGraphDataset, floorplan_collate_fn, is_adjacent
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from utils import bb_to_img, bb_to_vec, bb_to_seg, mask_to_bb, draw_graph
from PIL import Image, ImageDraw
from reconstruct import reconstructFloorplan
import svgwrite

from models import Generator
import networkx as nx
import matplotlib.pyplot as plt
from utils import ID_COLOR
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches - does not support larger batchs")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--with_boundary", action='store_true', default=True, help="include floorplan footprint")
parser.add_argument("--num_variations", type=int, default=10, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")

opt = parser.parse_args()
print(opt)

numb_iters = 200000
exp_name = 'exp_with_graph_global_new'
target_set = 'D'
checkpoint = './checkpoints/{}_{}_{}.pth'.format(exp_name, target_set, numb_iters)

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


def retrieve_connections(nodes, room_bb):
    edges = []
    nodes = [x for x in nodes.detach().cpu().numpy() if x >= 0]
    room_bb = room_bb.view(-1, 4).detach().cpu().numpy()
    for k, bb1 in enumerate(room_bb):
        for l, bb2 in enumerate(room_bb):
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
generator = Generator()
generator.load_state_dict(torch.load(checkpoint))
generator.eval()

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = '/local-scratch2/nnauata/autodesk/FloorplanDataset/'

# Configure data loader
rooms_path = '/local-scratch2/nnauata/autodesk/FloorplanDataset/'
fp_dataset = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), target_set=target_set, split='eval')
fp_loader = torch.utils.data.DataLoader(fp_dataset, 
                                        batch_size=opt.batch_size, 
                                        shuffle=True,
                                        num_workers=opt.n_cpu,
                                        collate_fn=floorplan_collate_fn)
fp_iter = tqdm(fp_loader, total=len(fp_dataset) // opt.batch_size + 1)

# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Generate samples
all_imgs = []
for i, batch in enumerate(fp_iter):
    if i > 64:
        break
        
    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch

    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds

    # Sample noise as generator input
    layouts_imgs_tensor = []
    
    # draw graph
    graph_img = draw_graph(nds.detach().cpu().numpy(), eds.detach().cpu().numpy(), 0, im_size=256)
    all_imgs.append(graph_img)
    
    # reconstruct
    for j in range(opt.num_variations):
        z_shape = [real_mks.shape[0], opt.latent_dim]
        z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))

        with torch.no_grad():
            gen_mks = generator(z, given_nds, given_eds)
            gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()])
            real_bbs = np.array([np.array(mask_to_bb(mk)) for mk in real_mks.detach().cpu()])
        real_nodes = np.where(given_nds.detach().cpu()==1)[-1]
    
        # draw boxes - filling
        comb_img = np.ones((256, 256, 3)) * 255
        comb_img = Image.fromarray(comb_img.astype('uint8'))
        dr = ImageDraw.Draw(comb_img)
        for bb in gen_bbs:
            dr.rectangle(tuple(bb*8.0), fill='beige')
            
        # draw boxes - outline
        for nd, bb in zip(real_nodes, gen_bbs):
            color = ID_COLOR[nd + 1]
            dr.rectangle(tuple(bb*8.0), outline=color, width=4)
                
        im_arr = torch.tensor(np.array(comb_img).transpose(2, 0, 1)/255.0).float()
        all_imgs.append(im_arr)    
all_imgs = torch.stack(all_imgs)
save_image(all_imgs, "layout_variations.png", nrow=11, normalize=False)
    
