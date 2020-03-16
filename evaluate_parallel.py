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
from utils import bb_to_img, bb_to_vec, bb_to_seg, mask_to_bb
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
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches - does not support larger batchs")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--with_boundary", action='store_true', default=True, help="include floorplan footprint")
parser.add_argument("--num_variations", type=int, default=10, help="number of variations")
parser.add_argument("--checkpoint", type=str, default='', help="destination folder") 
parser.add_argument("--target_set", type=str, default='A', help="which split to remove")
opt = parser.parse_args()

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
    nodes = [x for x in nodes if x >= 0]
    room_bb = room_bb.reshape((-1, 4))
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
generator.load_state_dict(torch.load(opt.checkpoint))
generator.eval()

# Initialize variables
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()

# Configure data loader
rooms_path = '/local-scratch4/nnauata/autodesk/FloorplanDataset/'
fp_dataset = FloorplanGraphDataset(rooms_path, transforms.Normalize(mean=[0.5], std=[0.5]), \
                                   target_set=opt.target_set, split='eval')
fp_loader = torch.utils.data.DataLoader(fp_dataset, 
                                        batch_size=opt.batch_size, 
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=floorplan_collate_fn)
fp_iter = tqdm(fp_loader, total=len(fp_dataset) // opt.batch_size + 1)

# Optimizers
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Generate samples
graphs = []
for i, batch in enumerate(fp_iter):

    # Unpack batch
    mks, nds, eds, nd_to_sample, ed_to_sample = batch

    # Configure input
    real_mks = Variable(mks.type(Tensor))
    given_nds = Variable(nds.type(Tensor))
    given_eds = eds

    # Sample noise as generator input
    layouts_imgs_tensor = []

    # reconstruct
    z_shape = [real_mks.shape[0], opt.latent_dim]
    z = Variable(Tensor(np.random.normal(0, 1, tuple(z_shape))))
        
    with torch.no_grad():
        gen_mks = generator(z, given_nds, given_eds)
        gen_bbs = np.array([np.array(mask_to_bb(mk)) for mk in gen_mks.detach().cpu()])/float(gen_mks.shape[-1])
        real_bbs = np.array([np.array(mask_to_bb(mk)) for mk in real_mks.detach().cpu()])
        
    real_nodes = np.where(given_nds.detach().cpu()==1)[-1]
    g_pred = retrieve_connections(real_nodes, gen_bbs)
    g_true = [real_nodes, eds.detach().cpu().numpy()]
    
    # build predicted graph
    G_pred = nx.Graph()
    colors_G = []
    
    for k, label in enumerate(g_pred[0]):
        _type = label+1 
        G_pred.add_nodes_from([(k, {'label':_type})])
        colors_G.append(ID_COLOR[_type])
    for k, l in g_pred[1]:
        G_pred.add_edges_from([(k, l)])

    # build true graph 
    G_true = nx.Graph()
    colors_H = []
    for k, label in enumerate(g_true[0]):
        _type = label+1 
        if _type >= 0:
            G_true.add_nodes_from([(k, {'label':_type})])
            colors_H.append(ID_COLOR[_type])
            
#     print(G_pred.size(), G_pred.number_of_edges())
#     print(G_true.size(), G_true.number_of_edges())
    
    for k, m, l in g_true[1]:
        if m > 0:
            G_true.add_edges_from([(k, l)])
    graphs.append([G_pred, G_true, i])
        
#     # DEBUG 
#     plt.figure()
#     pos = nx.spring_layout(G_pred)
#     nx.draw(G_pred, pos, node_size=1000, node_color=colors_G, font_size=0, font_weight='bold')
#     plt.tight_layout()
#     plt.savefig('./dump/{}_pred_graph.jpg'.format(i), format="PNG")
    
#     print(G_true)
#     plt.figure()
#     pos = nx.spring_layout(G_true)
#     nx.draw(G_true, pos, node_size=1000, node_color=colors_H, font_size=0, font_weight='bold')
#     plt.tight_layout()
#     plt.savefig('./dump/{}_true_graph.jpg'.format(i), format="PNG")
#     if i > 10:
#         break
    
MAX = 40
def run_parallel(graphs):
    # Compute in parallel
    from joblib import Parallel, delayed
    import multiprocessing
    import time
    import functools
    import threading
    def with_timeout(timeout):
        def decorator(decorated):
            @functools.wraps(decorated)
            def inner(*args, **kwargs):
                pool = multiprocessing.pool.ThreadPool(1)
                async_result = pool.apply_async(decorated, args, kwargs)
                try:
                    return async_result.get(timeout)
                except multiprocessing.TimeoutError:
                    G_pred, G_true, _id = args
                    print('timed out {}'.format(_id))
                    return None, None
            return inner
        return decorator
    @with_timeout(3000)
    def processInput(G_pred, G_true, _id):
        dists = [x for x in nx.optimize_graph_edit_distance(G_pred, G_true, upper_bound=MAX)]
        if len(dists) > 0:
            min_dist = np.min(dists)
        else:
            min_dist = MAX
        print(min_dist, _id)
        return min_dist, len(G_true.nodes())
    num_cores = multiprocessing.cpu_count()
    
    results = []
#     lower = 0
#     upper = 2500
    
    lower = 0
    upper = 1000
    
    results += Parallel(n_jobs=num_cores)(delayed(processInput)(G_pred, G_true, _id) for G_pred, G_true, _id in graphs[lower:upper])
        
    edit_dist_per_node = defaultdict(list)
    for min_dist, graph_len in results:
        if min_dist is not None:
            edit_dist_per_node[graph_len].append(min_dist)

    all_dists = []
    for n in edit_dist_per_node:
        print(n, len(edit_dist_per_node[n]), np.mean(edit_dist_per_node[n]))
        all_dists += edit_dist_per_node[n]
    print('range mean', np.mean(all_dists))
    print(lower, upper)
    return
run_parallel(graphs)