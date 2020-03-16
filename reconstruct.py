import cv2
import numpy as np
import sys
import csv
import copy
from utils import *
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import matplotlib.pyplot as plt
# from utils.intersections import doIntersect
from skimage import measure
from collections import defaultdict
import svgwrite

def snap_corners(cs, ths=[4, 8]):
    new_cs = np.array(cs)
    for th in ths:
        for i in range(len(new_cs)):
            x0, y0 = new_cs[i]
            x0_avg, y0_avg = [], []
            tracker = []
            for j in range(len(new_cs)):
                x1, y1 = new_cs[j]

                # horizontals
                if abs(x1-x0) <= th:
                    x0_avg.append(x1) 
                    tracker.append((j, 0))
                # verticals
                if abs(y1-y0) <= th:
                    y0_avg.append(y1)
                    tracker.append((j, 1))
            avg_vec = [np.mean(x0_avg), np.mean(y0_avg)]

            # set others
            for k, m in tracker:
                new_cs[k, m] = avg_vec[m]
    return new_cs

def compute_edges_mask(junctions, lines_on, width=2):
    im = Image.new('L', (256, 256))
    draw = ImageDraw.Draw(im)
    for j1, j2 in lines_on:
        x1, y1 = junctions[j1]
        x2, y2 = junctions[j2]
        draw.line((x1, y1, x2, y2), width=width, fill='white')
    return np.array(im) 

def _flood_fill(edge_mask, x0, y0, tag):
    new_edge_mask = np.array(edge_mask)
    nodes = [(x0, y0)]
    new_edge_mask[x0, y0] = tag
    while len(nodes) > 0:
        x, y = nodes.pop(0)
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (0 <= x+dx < new_edge_mask.shape[0]) and (0 <= y+dy < new_edge_mask.shape[0]) and (new_edge_mask[x+dx, y+dy] == 0):
                new_edge_mask[x+dx, y+dy] = tag
                nodes.append((x+dx, y+dy))
    return new_edge_mask

def fill_regions(edge_mask):
    edge_mask = edge_mask
    tag = 2
    for i in range(edge_mask.shape[0]):
        for j in range(edge_mask.shape[1]):
            if edge_mask[i, j] == 0:
                edge_mask = _flood_fill(edge_mask, i, j, tag)
                tag += 1
    return edge_mask

def extract_corners_and_edges(regions, graph, globalIndex):
    corner_set, edge_set = set(), set()
    rooms_type, _ = graph
    rooms_im = Image.new('RGB', (256, 256), 'white')
    dr = ImageDraw.Draw(rooms_im)
    for k in range(regions.shape[-1]):
        reg = (regions[:, :, k]*255).astype('uint8')
        if (rooms_type[k] >= 0) and (np.array(np.where(reg > 0)).shape[-1] > 0):
            ret, thresh = cv2.threshold(reg, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            pts = [tuple([c[0][0], c[0][1]]) for c in contours[0]]
            # save rooms
            color = ID_COLOR[rooms_type[k]+1]
            try:
                dr.polygon(pts, fill=color)
                x0, y0 = pts[-1]
                for l in range(len(pts)):
                    x1, y1 = pts[l]
                    corner_set.add((x1, y1))
                    e = make_order((x0, y0, x1, y1))
                    edge_set.add(e)
                    x0, y0 = pts[l]
            except:
                continue
                
    rooms_im.putalpha(200) 
    rooms_im.save('./rooms/{}_rooms.png'.format(globalIndex))
    return corner_set, edge_set

def update_regions(junctions, lines_on, globalIndex):
    edge_mask = compute_edges_mask(junctions, lines_on, width=2)
    region_mask = fill_regions(edge_mask)
    masks, boxes, labels = [], [], []
    inds = np.where((region_mask > 2) & (region_mask < 255))
    tags = set(region_mask[inds])
    for t in tags:
        m = np.zeros((256, 256))
        inds = np.array(np.where(region_mask == t))
        m[inds[0, :], inds[1, :]] = 1.0
        masks.append(m)
    masks = np.stack(masks)
    
    wrong_masks = np.array(Image.open('./rooms/{}_rooms.png'.format(globalIndex)))
    fixed_masks = np.array(Image.new('RGBA', (256, 256)))
    for m in masks:
        inds = np.where(m>0)
        wrong_pxs = wrong_masks[inds]
        count = defaultdict(int)
        for px in wrong_pxs:
            count[tuple(px)]+=1
        winner = sorted(count.items(), key=lambda x:x[1], reverse=True)[0]
        fixed_masks[inds] = np.array(winner[0])
    room_im_updated = Image.fromarray(fixed_masks)
    room_im_updated.save('./rooms/{}_rooms_updated.png'.format(globalIndex))
    return

def _format(corner_set, edge_set):
    lines_on = []
    junctions = list(corner_set)
    juncs_on = range(len(junctions))
    for (x0, y0, x1, y1) in list(edge_set):
        endpoints = []
        for k, c in enumerate(corner_set):
            if tuple([x0, y0]) == c:
                endpoints.append(k)
            if tuple([x1, y1]) == c:
                endpoints.append(k)
        lines_on.append(endpoints)
    return junctions, juncs_on, lines_on

def _suppress(junctions, juncs_on, lines_on, corner_dist_thresh=4):
    dists = np.zeros((len(juncs_on), len(juncs_on)))
    c_map = defaultdict(list)
    for k, j1 in enumerate(juncs_on):
        for l, j2 in enumerate(juncs_on):
            c1 = junctions[j1]
            c2 = junctions[j2]
            dists[k, l] = np.linalg.norm(np.array(c1)-np.array(c2))
            dists[l, k] = np.linalg.norm(np.array(c1)-np.array(c2))
        for l, j2 in enumerate(juncs_on):
            if (l != k) and (j1 not in c_map) and (dists[k, l] <= corner_dist_thresh):
                c_map[j2] = j1
    
    new_lines_on = []
    for j1, j2 in lines_on:
        line = []
        if j1 in c_map:
            line.append(c_map[j1])
        else:
            line.append(j1)
            
        if j2 in c_map:
            line.append(c_map[j2])
        else:
            line.append(j2)
        new_lines_on.append(line)
    lines_on = new_lines_on
    juncs_on = [j for j in juncs_on if j not in c_map.keys()]
    return junctions, juncs_on, lines_on 

def draw_floorplan(dwg, junctions, juncs_on, lines_on):

    # draw edges
    for k, l in lines_on:
        x1, y1 = np.array(junctions[k])
        x2, y2 = np.array(junctions[l])
        #fill='rgb({},{},{})'.format(*(np.random.rand(3)*255).astype('int'))
        dwg.add(dwg.line((float(x1), float(y1)), (float(x2), float(y2)), stroke='black', stroke_width=4, opacity=1.0))

    # draw corners
    for j in juncs_on:
        x, y = np.array(junctions[j])
        dwg.add(dwg.circle(center=(float(x), float(y)), r=3, stroke='red', fill='white', stroke_width=2, opacity=1.0))
    return 

def getIntersection(region_map, j1, j2, width=1):
    x1, y1 = j1
    x2, y2 = j2
    m = Image.new('L', (256, 256))
    dr = ImageDraw.Draw(m)
    dr.line((x1, y1, x2, y2), width=width, fill='white')
    inds = np.array(np.where(np.array(m) > 0.0))
    return np.logical_and(region_map, m).sum()/inds.shape[1]

def getAngle(pt1, pt2):
    # return angle in clockwise direction
    x, y = pt1
    xn, yn = pt2
    dx, dy = xn-x, yn-y
    dir_x, dir_y = (dx, dy)/(np.linalg.norm([dx, dy])+1e-8)
    rad = np.arctan2(-dir_y, dir_x)
    ang = np.degrees(rad)
    if ang < 0:
        ang = (ang + 360) % 360
    return 360-ang

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

def corner_intersection(junctions, juncs_on, j1, j2):
    x1, y1 = junctions[j1]
    x2, y2 = junctions[j2]
    for j3 in juncs_on:
        if (j1 != j3) and (j2 != j3):
            x3, y3 = junctions[j3]
            corner_im = Image.new('L', (256, 256))
            dr = ImageDraw.Draw(corner_im)
            dr.rectangle((x3-2, y3-2, x3+2, y3+2), fill='white')
            corner_im = np.array(corner_im)/255.0
            weight = getIntersection(corner_im, junctions[j1], junctions[j2], width=1)
            if weight > 0:
                return True
    return False

def clean_corners_and_edges(junctions, juncs_on, lines_on):
    
    # fix colinearity
    glob_index = 0
    edge_mask = compute_edges_mask(junctions, lines_on, width=4)
    edge_mask[edge_mask>0]=1.0
    new_lines_on = []
    deb_edge = Image.fromarray(edge_mask*255.0).convert('RGB')
    deb_edge.save('./dump/edge_mask.jpg')
    
    # remove duplicated corners
    junctions = set([(int(x0), int(y0)) for x0, y0 in junctions])
    junctions = list(junctions)
    juncs_on = list(range(len(junctions)))
    for k, j1 in enumerate(juncs_on):
        for l, j2 in enumerate(juncs_on):
            if k > l:
                edge_interc = getIntersection(edge_mask, junctions[j1], junctions[j2], width=1)
                x1, y1 = junctions[j1]
                x2, y2 = junctions[j2]
                
 
                
#                 deb = Image.new('RGB', (256, 256))
#                 dr = ImageDraw.Draw(deb)
#                 dr.line((x1, y1, x2, y2), width=1, fill='white')
#                 deb.save('./dump/{}_{}.jpg'.format(glob_index, weight))
#                 glob_index += 1
#                 if glob_index > 100:
#                     exit(0)
                    
                if (edge_interc == 1.0) and (corner_intersection(junctions, juncs_on, j1, j2)==False):
                    new_lines_on.append((j1, j2))
   
    # remove bad corners
    count = defaultdict(int)
    for j1, j2 in new_lines_on:
        count[j1] +=1
        count[j2] +=1
    
    new_juncs_on = []
    for j1, j2 in new_lines_on:
        if j1 not in new_juncs_on:
            new_juncs_on.append(j1)
        if j2 not in new_juncs_on:
            new_juncs_on.append(j2)
        
    return junctions, new_juncs_on, new_lines_on

def reconstructFloorplan(regions, graph, globalIndex):    
    corner_set, edge_set = extract_corners_and_edges(regions, graph, globalIndex)
    junctions, juncs_on, lines_on = _format(corner_set, edge_set)
    junctions, juncs_on, lines_on = _suppress(junctions, juncs_on, lines_on)
    junctions = snap_corners(junctions)
    junctions, juncs_on, lines_on = clean_corners_and_edges(junctions, juncs_on, lines_on)
    juncs_on, lines_on = remove_junctions(junctions, juncs_on, lines_on)
    update_regions(junctions, lines_on, globalIndex) 
    return junctions, juncs_on, lines_on

##########################################################################################################
############################################ HELPER FUNCTIONS ############################################
##########################################################################################################
def make_order(e):
	x0, y0, x1, y1 = e
	if x1 < x1 or y0 < y1:
		return (x0, y0, x1, y1)
	else:
		return (x1, y1, x0, y0)