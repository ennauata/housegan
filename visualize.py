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
from input_graphs import *
from models import Generator

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=20, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--img_size", type=int, default=4, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--with_boundary", action='store_true', default=True, help="include floorplan footprint")
parser.add_argument("--num_variations", type=int, default=10, help="number of variations")
parser.add_argument("--exp_folder", type=str, default='exp', help="destination folder")
parser.add_argument("--checkpoint", type=str, default='checkpoints/gen_neighbour_exp_10_nodes_train_split_1000000.pth', help="destination folder")
opt = parser.parse_args()
print(opt)

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

# Create folder
os.makedirs(opt.exp_folder, exist_ok=True)

# Initialize generator and discriminator
generator = Generator(opt.with_boundary)
generator.load_state_dict(torch.load(opt.checkpoint))
generator.eval()

# Initialize variables
img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
rooms_path = '/local-scratch2/nnauata/autodesk/CubiCasa5k/'

# Initialize dataset iterator
fp_dataset = FloorplanGraphDataset(rooms_path, split='test')
fp_loader = torch.utils.data.DataLoader(fp_dataset, 
                                        batch_size=opt.batch_size, 
                                        shuffle=True, collate_fn=floorplan_collate_fn)
# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ------------
#  Vectorize
# ------------
globalIndex = 0
# np.random.seed(100)

# Generate Mod 1
boundary_bb = Variable(torch.tensor(in_boundary_1).type(Tensor))
nodes = torch.tensor(in_nodes_1)
triples = torch.tensor(in_triples_1)
room_to_sample = torch.tensor(np.zeros((nodes.shape[0])))
triple_to_sample = torch.tensor(np.zeros((triples.shape[0])))
z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
gen_room_bb = generator(z, [nodes, triples], room_to_sample, boundary=boundary_bb)
gen_imgs_tensor_1 = bb_to_img(gen_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample)
print(z)

# Generate Mod 2
boundary_bb = Variable(torch.tensor(in_boundary_2).type(Tensor))
nodes = torch.tensor(in_nodes_2)
triples = torch.tensor(in_triples_2)
room_to_sample = torch.tensor(np.zeros((nodes.shape[0])))
triple_to_sample = torch.tensor(np.zeros((triples.shape[0])))
# z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
gen_room_bb = generator(z, [nodes, triples], room_to_sample, boundary=boundary_bb)
gen_imgs_tensor_2 = bb_to_img(gen_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample)
print(z)

# Generate Mod 3
boundary_bb = Variable(torch.tensor(in_boundary_3).type(Tensor))
nodes = torch.tensor(in_nodes_3)
triples = torch.tensor(in_triples_3)
room_to_sample = torch.tensor(np.zeros((nodes.shape[0])))
triple_to_sample = torch.tensor(np.zeros((triples.shape[0])))
# z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
gen_room_bb = generator(z, [nodes, triples], room_to_sample, boundary=boundary_bb)
gen_imgs_tensor_3 = bb_to_img(gen_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample)
print(z)

# Generate Mod 4
boundary_bb = Variable(torch.tensor(in_boundary_4).type(Tensor))
nodes = torch.tensor(in_nodes_4)
triples = torch.tensor(in_triples_4)
room_to_sample = torch.tensor(np.zeros((nodes.shape[0])))
triple_to_sample = torch.tensor(np.zeros((triples.shape[0])))
# z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
gen_room_bb = generator(z, [nodes, triples], room_to_sample, boundary=boundary_bb)
gen_imgs_tensor_4 = bb_to_img(gen_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample)
print(z)

# Generate Mod 5
boundary_bb = Variable(torch.tensor(in_boundary_5).type(Tensor))
nodes = torch.tensor(in_nodes_5)
triples = torch.tensor(in_triples_5)
room_to_sample = torch.tensor(np.zeros((nodes.shape[0])))
triple_to_sample = torch.tensor(np.zeros((triples.shape[0])))
# z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
gen_room_bb = generator(z, [nodes, triples], room_to_sample, boundary=boundary_bb)
gen_imgs_tensor_5 = bb_to_img(gen_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample)
print(z)

# Generate Mod 6
boundary_bb = Variable(torch.tensor(in_boundary_6).type(Tensor))
nodes = torch.tensor(in_nodes_6)
triples = torch.tensor(in_triples_6)
room_to_sample = torch.tensor(np.zeros((nodes.shape[0])))
triple_to_sample = torch.tensor(np.zeros((triples.shape[0])))
# z = Variable(Tensor(np.random.normal(0, 1, (1, opt.latent_dim))))
gen_room_bb = generator(z, [nodes, triples], room_to_sample, boundary=boundary_bb)
gen_imgs_tensor_6 = bb_to_img(gen_room_bb.data, [nodes, triples], room_to_sample, triple_to_sample)
print(z)

# Save all images
all_imgs = torch.cat([gen_imgs_tensor_1, gen_imgs_tensor_2, gen_imgs_tensor_3, gen_imgs_tensor_4, gen_imgs_tensor_5, gen_imgs_tensor_6])
save_image(all_imgs, "./exp_mod/test_1.png", nrow=4)

exit(0)