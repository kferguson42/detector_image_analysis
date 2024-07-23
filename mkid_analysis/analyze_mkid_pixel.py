import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os
from skimage import filters, transform, io, morphology, color, feature, draw, segmentation
from copy import deepcopy
import argparse as ap
from PIL import Image

# for the following imports to work, you must add the repository to your PYTHONPATH (see README)
import geometry_utils as geo
import image_utils
from spt4.make_capacitor_template import draw_capacitor_lines
from spt4.spt4_mapping_utils import image_num_to_capacitor_info
import spt4.analysis_params as aparams
import spt4.plot_functions as pf

Image.MAX_IMAGE_PIXELS = None

'''
This script reads in an image of an SPT-4 mKID bolometer, determines
the location of the mKID in the image, and searches for breaks in the
inductor to predict whether the detector will be operatable. Then it
measures the width of the inductor lines relative to the design.
Finally, it repeats the process for the capacitors on a pixel.
'''

p = ap.ArgumentParser()
p.add_argument('-i', '--image', type=str, default='/home/kferguson/DetectorAnalysis/spt4_detector_files/',
               help="Can be either a single file or a directory.")
p.add_argument('-m', '--magnification', type=int, default=20)
p.add_argument('-s', '--save-dir', type=str, default='/home/kferguson/Analyses/DetectorAnalysis/spt4_detector_files/results/updated_stitching/')
p.add_argument('--aux-dir', type=str, default='/home/kferguson/Analyses/DetectorAnalysis/spt3g_wafer_ML/spt4/')
p.add_argument('-t', '--threshold', type=float, default=0.2,
               help="Normalized value above which to set Scharr edges to 1 (and below to 0) before skeletonization")
p.add_argument('--index', type=int, default=0,
               help="Detector indexing number (used to name output files)")
p.add_argument('--seed', type=int, default=-1,
               help="Random number generator seed.  Optional, will use the value of 'index' if not provided.")
p.add_argument('--sim', default=False, action='store_true',
               help="Should be True if the image you're running is a simulated one.  The only difference is that these images are already"
               "greyscale, so don't need to be converted as such.")
args = p.parse_args()

# goofy and probably bad-practice workaround to give all the plotting functions access to some
# common script-level variables that they need
pf.save_dir = args.save_dir
pf.index = args.index

# probabilistic Hough transform has an element of randomness, set seed to make results reproducible
if args.seed == -1:
    args.seed = args.index
np.random.seed(args.seed)

# get parameters from analysis_params.py
if args.magnification == 50: # here for historical reasons, shouldn't actually be using 50x images
    params = aparams.params_50x
elif args.magnification == 20:
    params = aparams.params_20x
line_frac = params['line_frac']
hough_thresh = params['hough_thresh']
min_length = params['min_length']
gap = params['gap']
angs = params['angs']
mask_frac = params['mask_frac']
scaling = params['scaling']
seed_locs = params['seed_locs']
cutout_size = params['cutout_size']
microns_per_pixel_x = params['microns_per_pixel_x']
microns_per_pixel_y = params['microns_per_pixel_y']

# create output directory if it doesn't exist
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# load in image
# cut out very edge pixels to remove stitching boundaries; they'll screw up the thresholding on the edge detection
f = io.imread(args.image)[100:-100, 100:-100]
if args.sim:
    y,x = f.shape
    image = deepcopy(f)
else:
    y,x,_ = f.shape
    image = color.rgb2gray(f)
imshape = (image.shape[1], image.shape[0])
xx = x/2
yy = y/2
pf.x = x; pf.y = y
cutoff_x = (x - cutout_size) // 2
cutoff_y = (y - cutout_size) // 2

# filter, skeletonize image
edges = filters.gaussian(image, sigma=1)
edges = filters.scharr(edges)
edges /= np.max(edges)
e = deepcopy(edges)
e[e<args.threshold] = 0
e[e>=args.threshold] = 1
skeleton = morphology.skeletonize(e)
# try to get rid of small dust motes
skeleton = morphology.remove_small_objects(skeleton, 10, connectivity=2)

thresh_image = filters.threshold_otsu(image[y//2-4000:y//2+4000, x//2-4000:x//2+4000])
binary = image > thresh_image

# find general bolometer island location in image
# first get orientation angle up to 90-degree modulus
lines = transform.probabilistic_hough_line(skeleton[y//2-y//line_frac:y//2+y//line_frac, x//2-x//line_frac:x//2+x//line_frac],
                                           threshold=hough_thresh, line_length=min_length, line_gap=gap, theta=angs, seed=args.seed)
ang, km = geo.hough_lines_to_angle(lines)

# create template of inductor geometry in each of the four possible orientations
s = {'regular':np.sin(ang), '+90':np.sin(ang+(np.pi/2)), '+180':np.sin(ang+np.pi), '-90':np.sin(ang-(np.pi/2))}
c = {'regular':np.cos(ang), '+90':np.cos(ang+(np.pi/2)), '+180':np.cos(ang+np.pi), '-90':np.cos(ang-(np.pi/2))}
templates = {'regular':np.zeros(image.shape, dtype=np.uint8), '+90':np.zeros(image.shape, dtype=np.uint8),
             '+180':np.zeros(image.shape, dtype=np.uint8), '-90':np.zeros(image.shape, dtype=np.uint8)}
fit_mask = {'regular':[], '+90':[], '+180':[], '-90':[]}
rot_angles = {'regular':0, '+90':np.pi/2., '+180':np.pi, '-90':-np.pi/2.}
mask_file = args.aux_dir + 'pixel_mask_physical_units.pkl'
mask_lines = pk.load(open(mask_file, 'rb'))
for l in mask_lines:
    for o in templates.keys():
        p1 = geo.rotate_point(l[0], ang + rot_angles[o], scaling/microns_per_pixel_x,
                              scaling/microns_per_pixel_y, (xx, yy))
        p2 = geo.rotate_point(l[1], ang + rot_angles[o], scaling/microns_per_pixel_x,
                              scaling/microns_per_pixel_y, (xx, yy))
        row, col, val = draw.line_aa(int(np.round(p1[1])), int(np.round(p1[0])),
                                 int(np.round(p2[1])), int(np.round(p2[0])))
        templates[o][row, col] = val * 255
        fit_mask[o].append([p1, p2])

# perform matched filter between each template and image to determine correct orientation (and also location)
filtered = {}
maxs = {}
centers = {}
for o in templates.keys():
    templates[o] = templates[o][y//2-y//mask_frac:y//2+y//mask_frac, x//2-x//mask_frac:x//2+x//mask_frac]
    filtered[o] = feature.match_template(skeleton, templates[o], pad_input=True)
    filtered[o] = filters.gaussian(filtered[o], sigma=2.5)
    maxs[o] = np.max(filtered[o])
    centers[o] = np.unravel_index(np.argmax(filtered[o]), filtered[o].shape)

orientation = list(templates.keys())[np.argmax(list(maxs.values()))]
fit_loc = np.unravel_index(np.argmax(filtered[orientation]), filtered[orientation].shape)
x_loc, y_loc = fit_loc[::-1]
center_coord = [x_loc, y_loc]
diff = np.array([x_loc - xx, y_loc - yy])
fit_mask[orientation] += diff
pf.x_loc = x_loc; pf.y_loc = y_loc

# generate pixel locations of flood fill seeds. we will do this again later after we refine
# the angle, but we need this initial estimate to actually *get* that refined angle; the
# cross-correlation works a lot better if things are filled in than if it's working on a skeleton.
# technically, we could get around this by using draw.polygon below where we are drawing the
# template, but it's much faster to use draw.line and just flood-fill the template in.
seed_image_locs = []
for pt in seed_locs:
    new_pt = geo.rotate_point(pt, ang + rot_angles[orientation], scaling, scaling, (xx, yy))
    new_pt = np.round(new_pt + diff)
    new_pt = np.array(new_pt, dtype=int)
    seed_image_locs.append((new_pt[0]-cutoff_x, new_pt[1]-cutoff_y))

# angle found earlier is a bit inaccurate, try a range of angles around it to refine
ang_spread = np.linspace(-1.*(np.pi/180.), 1.*(np.pi/180.), 100)
extrema = {'left':[4043], 'right':[4068], 'vertical':[3139,11633]}
plot_lines = {}
xcorrs = []

for delta in ang_spread:
    temp = np.zeros(image.shape)
    new_ang = delta+ang+rot_angles[orientation]
    plot_lines[delta] = []
    pts = []
    for l in mask_lines:
        p1 = geo.rotate_point(l[0], new_ang, scaling/microns_per_pixel_x,
                              scaling/microns_per_pixel_y, (xx+diff[0], yy+diff[1]))
        p2 = geo.rotate_point(l[1], new_ang, scaling/microns_per_pixel_x,
                              scaling/microns_per_pixel_y, (xx+diff[0], yy+diff[1]))
        row, col = draw.line(int(np.round(p1[1])), int(np.round(p1[0])),
                             int(np.round(p2[1])), int(np.round(p2[0])))
        temp[row, col] = 1
        plot_lines[delta].append([p1, p2])
    temp = temp[cutoff_y:-cutoff_y,cutoff_x:-cutoff_x]
    for ind in range(3):
        segmentation.flood_fill(temp, seed_image_locs[ind][::-1], True, connectivity=1, in_place=True)
    xcorrs.append(image_utils.xcorr_same_size(temp, binary[cutoff_y:-cutoff_y,cutoff_x:-cutoff_x]))

delta_ang = ang_spread[np.argmax(xcorrs)]
final_rot_angle = delta_ang + ang + rot_angles[orientation]
fit_mask[orientation] = np.array(plot_lines[delta_ang])
split_mask = [fit_mask[orientation][:16227], fit_mask[orientation][16227:]]

c = np.cos(delta_ang+ang+rot_angles[orientation])
s = np.sin(delta_ang+ang+rot_angles[orientation])

# Now with refined angle, do real flood fill
# Cut off inductors so flood fill doesn't go to capacitors
skel = deepcopy(skeleton)
extrema = {'left':[4043], 'right':[4068], 'vertical':[3139,11633]}
for obj in ['left', 'right', 'vertical']:
    edge_lines = pk.load(open(args.aux_dir + '%s_meander_physical_units.pkl'%(obj), 'rb'))
    for ind in extrema[obj]:
        # pad line a bit to make sure there are no gaps
        l = geo.lengthen_line_segment(edge_lines[ind], 50.0*microns_per_pixel_x) # doesn't matter which scaling used, use x
        p1 = geo.rotate_point(l[0], final_rot_angle, scaling/microns_per_pixel_x,
                              scaling/microns_per_pixel_y, (xx+diff[0], yy+diff[1]))
        p2 = geo.rotate_point(l[1], final_rot_angle, scaling/microns_per_pixel_x,
                              scaling/microns_per_pixel_y, (xx+diff[0], yy+diff[1]))
        rr, cc, = draw.line(int(np.round(p1[1])), int(np.round(p1[0])),
                            int(np.round(p2[1])), int(np.round(p2[0])))
        skel[rr, cc] = 1

# Flood fill algorithm to find breaks
cutoff_x = (x - cutout_size) // 2
cutoff_y = (y - cutout_size) // 2
seed_image_locs = []
for pt in seed_locs:
    new_pt = geo.rotate_point(pt, final_rot_angle, scaling, scaling, (xx, yy))
    new_pt = np.round(new_pt + diff)
    new_pt = np.array(new_pt, dtype=int)
    seed_image_locs.append((new_pt[0]-cutoff_x, new_pt[1]-cutoff_y))

filled = segmentation.flood_fill(skel[cutoff_y:-cutoff_y,cutoff_x:-cutoff_x], seed_image_locs[0][::-1], True, connectivity=1)
segmentation.flood_fill(filled, seed_image_locs[1][::-1], True, connectivity=1, in_place=True)

# draw a circle in an empty region and count filled pixels to determine if there has been
# overspill from the flood fill
circ_centers = [(250, 250), (0, 630), (0, -630)]
plot_temp = deepcopy(filled); plot_temp = np.array(plot_temp, dtype=float)
overspill_nums = []; overspill_tots = []
for circ_center in circ_centers:
    rot_cir_center = geo.rotate_point(circ_center, final_rot_angle, scaling/microns_per_pixel_x,
                                      scaling/microns_per_pixel_y, (xx+diff[0]-cutoff_x, yy+diff[1]-cutoff_y))
    rot_cir_center = (int(np.round(rot_cir_center[0])), int(np.round(rot_cir_center[1])))
    rr, cc = draw.circle_perimeter(rot_cir_center[1], rot_cir_center[0],
                                   radius=int(np.round(40*(scaling/microns_per_pixel_x))), shape=filled.shape)
    temp = np.zeros(filled.shape)
    temp[rr, cc] = 0.5
    plot_temp[rr, cc] = 0.5
    segmentation.flood_fill(temp, rot_cir_center[::-1], 1, connectivity=1, in_place=True)
    overspill_nums.append(np.sum(filled[temp == 1]))
    overspill_tots.append(np.sum(temp[temp == 1]))

# plot to make sure circle is in reasonable place
pf.inductor_overspill_fig(plot_temp, overspill_nums, overspill_tots)

# template subtraction; not really used for much, can maybe delete
rotated_template_arr = np.zeros(image.shape)
split_rotated_template_arr = [np.zeros(image.shape), np.zeros(image.shape)]
for obj in ['left', 'right', 'vertical']:
    edge_lines = pk.load(open(args.aux_dir + '%s_meander_physical_units.pkl'%(obj), 'rb'))
    pts = []
    for l in edge_lines:
        p1 = geo.rotate_point(l[0], final_rot_angle, scaling/microns_per_pixel_x,
                              scaling/microns_per_pixel_y, (0, 0))
        pts.append(p1)
    pts = np.array(pts)
    pts[:,0] += x_loc; pts[:,1] += y_loc
    rr, cc = draw.polygon(pts[:,1], pts[:,0], rotated_template_arr.shape)
    rotated_template_arr[rr, cc] = 1
    if obj == 'vertical':
        split_rotated_template_arr[0][rr, cc] = 1
    else:
        split_rotated_template_arr[1][rr, cc] = 1

thresh_image = filters.threshold_otsu(image[y//2-4000:y//2+4000, x//2-4000:x//2+4000])
binary = image > thresh_image

# measure detector line width, looking in multiple "good" regions and averaging
ratios = [[],[]]
img_nums = [[], []]
thr_nums = [[], []]
box_nums = [[], []]
measure_bounds = {'long' : lambda i : [(-20,-400+i*40), (-20,-360+i*40), (20,-360+i*40), (20,-400+i*40)],
                  'broken' : lambda i : [(-420+i*40+(i//10)*40, -20), (-380+i*40+(i//10)*40, -20), (-380+i*40+(i//10)*40, 20), (-420+i*40+(i//10)*40, 20)]
                  }
for i, inductor in enumerate(['long', 'broken']):
    for j in range(20):
        image_bounds = []
        for p1 in measure_bounds[inductor](j):
            p1 = geo.rotate_point(p1, final_rot_angle, scaling/microns_per_pixel_x,
                                  scaling/microns_per_pixel_y, (x_loc, y_loc))
            image_bounds.append(p1)
        image_bounds = np.array(image_bounds)
        row, col = draw.polygon(image_bounds[:,1], image_bounds[:,0])
        img_num = np.sum(binary[row, col])
        thr_num = len(np.where(rotated_template_arr[row, col]==1)[0])
        img_nums[i].append(img_num)
        thr_nums[i].append(thr_num)
        box_nums[i].append(len(row))
        ratios[i].append(img_num / thr_num)
pf.template_subtraction_fig(image, binary, rotated_template_arr, ratios)

# count percentage of inductor that was filled in
rotated_template_arr_skel = morphology.skeletonize(rotated_template_arr[cutoff_y:-cutoff_y, cutoff_x:-cutoff_x])
split_rotated_template_arr_skel = {}
for j in range(2):
    split_rotated_template_arr_skel[j] = morphology.skeletonize(split_rotated_template_arr[j][cutoff_y:-cutoff_y, cutoff_x:-cutoff_x])
filled_num_long = np.sum(filled[split_rotated_template_arr_skel[0]==1])
total_num_long = np.sum(split_rotated_template_arr_skel[0])
filled_num_broken = np.sum(filled[split_rotated_template_arr_skel[1]==1])
total_num_broken = np.sum(split_rotated_template_arr_skel[1])

# plot major steps of inductor analysis
pf.main_inductor_fig(image, skel, filtered[orientation], filled, lines, fit_mask[orientation],
                     km, seed_image_locs, cutoff_x, cutoff_y, maxs[orientation],
                     filled_num_long/total_num_long, filled_num_broken/total_num_broken)

# -----------------------------------------------------
# NOW FOR ALL OF THE CODE THAT DEALS WITH THE CAPACITOR
# -----------------------------------------------------

# get info for this particular MKID pixel
start1, span1, trim1, start2, span2, trim2 = image_num_to_capacitor_info(args.index, 'spt4_dense_chip', trim=True)
starting_angles = [-1. * (delta_ang+ang+rot_angles[orientation]) + start1 + np.pi, -1. * (delta_ang+ang+rot_angles[orientation]) + start2 + np.pi]
angle_spans = [span1, span2]
trim_angles = [trim1, trim2]
radius_adjustments = aparams.cap_design_params['radius_adjustments']
cap_measure_bounds = [(680,-10), (680,10), (930,10), (930,-10)]
cap_measure_angs = [np.linspace(np.pi/2-starting_angles[0]-2*(np.pi/180), np.pi/2-starting_angles[0]-span1+2*(np.pi/180), num=16),
                    np.linspace(np.pi/2-starting_angles[1]-2*(np.pi/180), np.pi/2-starting_angles[1]-span2+2*(np.pi/180), num=16)]

cap_filled = {}
cap_flagged = {}
cap_filled_legs = {}
cap_flagged_break = {}
cap_flagged_bridge = {}
cap_overspill_num = {}
cap_overspill_tot = {}
cap_ratios = {}
cap_translations = {}

for i in range(2):
    # iterate over the two capacitors on each pixel
    pf.i = i
    start = starting_angles[i]
    span = angle_spans[i]
    trim_angle = trim_angles[i]
    new_lines = []
    split_lines = []
    cap_rotated_template_arr = np.zeros(image.shape)
    delta_rad = np.linspace(-2, 2, 101)
    cap_maxs = []
    cap_template = np.zeros(image.shape)

    cap_seed_locs = [(-3, 800), (3, 800)]
    seed_image_locs = []
    seed_angs = [-start, -start-span]
    co = [np.cos(-start), np.cos(-start-span)]
    si = [np.sin(-start), np.sin(-start-span)]

    # generate capacitor lines (must be done this way instead of just loading in a pickle file like
    # with the inductors because each capacitor has a different angular span and trim value)
    for invert in [False, True]:
        cap_lines = draw_capacitor_lines(start, span, invert_dir=invert,
                                         outer_radius=aparams.cap_design_params['outer_radius'] + radius_adjustments[i],
                                         trim_angle=trim_angle)
        scaled_lines = []
        for l in cap_lines:
            p1 = geo.rotate_point((l[0][0], l[1][0]), 0, scaling/microns_per_pixel_x,
                                  scaling/microns_per_pixel_y, (xx+diff[0], yy+diff[1]))
            p2 = geo.rotate_point((l[0][1], l[1][1]), 0, scaling/microns_per_pixel_x,
                                  scaling/microns_per_pixel_y, (xx+diff[0], yy+diff[1]))
            scaled_lines.append((p1, p2))

            row, col, val = draw.line_aa(int(np.round(p1[1])), int(np.round(p1[0])),
                                         int(np.round(p2[1])), int(np.round(p2[0])))
            cap_template[row, col] = val * 255

        new_lines.extend(scaled_lines)
        split_lines.append(scaled_lines)

    min_x = int(np.min(np.array(new_lines)[:,0,0])) - 50
    max_x = int(np.max(np.array(new_lines)[:,0,0])) + 50
    min_y = int(np.min(np.array(new_lines)[:,0,1])) - 50
    max_y = int(np.max(np.array(new_lines)[:,0,1])) + 50

    # do flood fill on capacitors
    new_skel = deepcopy(skeleton)
    # add lines so flood fill doesn't go back into inductors
    cutoff_inds = [1, 3, 6405, 6407] #assuming each arc has 200 lines, as currently coded
    for ind in cutoff_inds:
        l = geo.lengthen_line_segment(new_lines[ind], 50.0)
        p1 = l[0]
        p2 = l[1]
        rr, cc = draw.line(int(np.round(p1[1])), int(np.round(p1[0])),
                           int(np.round(p2[1])), int(np.round(p2[0])))
        new_skel[rr, cc] = 1

    cap_rotated_template_arr = np.zeros(image.shape)
    filled_nums = [[], []]; total_nums = [[], []]    
    for j in range(2): # iterate over two sides of capacitor
        # count filled pixel fraction in side we're filling up
        temp = np.zeros(cap_rotated_template_arr.shape)
        pts = []
        for l in split_lines[j]:
            pts.append(l[0])
        pts = np.array(pts)
        rr, cc = draw.polygon(pts[:,1], pts[:,0], cap_rotated_template_arr.shape)
        temp[rr, cc] = 1
        cap_rotated_template_arr[temp==1] = 1

    cap_rotated_template_arr_skel = morphology.skeletonize(cap_rotated_template_arr)
    #pf.capacitor_template_fit(half_filled, cap_rotated_template_arr_skel, seed_image_locs, filled_nums, total_nums, adjustment=False)

    # capacitor template needs to be shifted slightly to line up better with image; find best translation here
    cap_rotated_template_arr = cap_rotated_template_arr[min_y:max_y, min_x:max_x]
    search_area = 20
    xcorrs = image_utils.xcorr(cap_rotated_template_arr, binary[min_y:max_y, min_x:max_x], np.array([int((max_y-min_y)/2), int((max_x-min_x)/2)]), search_area)
    cap_diff = np.unravel_index(np.argmax(xcorrs), xcorrs.shape) - np.array([search_area/2, search_area/2])
    cap_diff = cap_diff[::-1]
    cap_translations[i] = [cap_diff[0], cap_diff[1]]

    # Flood fill algorithm to find breaks
    for j, pt in enumerate(cap_seed_locs):
        new_pt = geo.rotate_point(pt, seed_angs[j], scaling/microns_per_pixel_x,
                                  scaling/microns_per_pixel_y, (xx, yy))
        new_pt = np.round(new_pt + diff + cap_diff)
        new_pt = np.array(new_pt, dtype=int)
        seed_image_locs.append((new_pt[0], new_pt[1]))
    filled = segmentation.flood_fill(new_skel, seed_image_locs[0][::-1], True, connectivity=1)
    half_filled = [deepcopy(filled)]
    half_filled.append(segmentation.flood_fill(new_skel, seed_image_locs[1][::-1], True, connectivity=1))
    segmentation.flood_fill(filled, seed_image_locs[1][::-1], True, connectivity=1, in_place=True)

    # now do flood fill accounting again with the adjusted template location
    cap_rotated_template_arr = np.zeros(image.shape)
    filled_nums = [[], []]; total_nums = [[], []]
    cap_overspill_num[i] = {}
    cap_overspill_tot[i] = {}
    for j in range(2): # iterate over two sides of capacitor
        # count filled pixel fraction in side we're filling up
        temp = np.zeros(cap_rotated_template_arr.shape)
        pts = []
        for l in split_lines[j]:
            pts.append(l[0] + cap_diff)
        pts = np.array(pts)
        rr, cc = draw.polygon(pts[:,1], pts[:,0], cap_rotated_template_arr.shape)
        temp[rr, cc] = 1
        cap_rotated_template_arr[temp==1] = 1
        temp = morphology.skeletonize(temp)
        filled_nums[0].append(np.sum(half_filled[j][temp==1]))
        total_nums[0].append(np.sum(temp))

        # count filled pixel fraction in side we're leaving alone (in order to see if there's a short)
        filled_nums[1].append(np.sum(half_filled[j-1][temp==1]))
        total_nums[1].append(np.sum(temp))
        del temp

        # draw a circle in an empty region and count filled pixels to determine if there has been
        # overspill from the flood fill
        circ_center = (250, 250)
        rot_cir_center = geo.rotate_point(circ_center, final_rot_angle, scaling/microns_per_pixel_x,
                                          scaling/microns_per_pixel_y, (xx+diff[0], yy+diff[1]))
        rot_cir_center = (int(np.round(rot_cir_center[0])), int(np.round(rot_cir_center[1])))
        rr, cc = draw.circle_perimeter(rot_cir_center[1], rot_cir_center[0],
                                       radius=int(np.round(50*(scaling/microns_per_pixel_x))), shape=filled.shape)
        temp = np.zeros(half_filled[j].shape)
        temp[rr, cc] = 1
        segmentation.flood_fill(temp, rot_cir_center[::-1], 1, connectivity=1, in_place=True)
        cap_overspill_num[i][j] = np.sum(filled[temp == 1])
        cap_overspill_tot[i][j] = np.sum(temp[temp == 1])
        # plot to make sure circle is in reasonable place
        temp = deepcopy(half_filled[j])
        temp = np.array(temp, dtype=float)
        temp[rr, cc] = 0.5
        pf.capacitor_overspill_fig(temp, cap_overspill_num, cap_overspill_tot, j)

    cap_rotated_template_arr_skel = morphology.skeletonize(cap_rotated_template_arr)
    cap_filled_num = np.sum(filled[cap_rotated_template_arr_skel==1])
    cap_total_num = np.sum(cap_rotated_template_arr_skel)

    pf.capacitor_template_fit(half_filled, cap_rotated_template_arr_skel, seed_image_locs,
                              filled_nums, total_nums, adjustment=True, cap_diff=cap_diff)
    pf.capacitor_xcorr_fig(xcorrs)
    pf.capacitor_skeleton_fig(new_skel, new_lines, seed_image_locs)
    pf.capacitor_flood_fill(filled, cap_rotated_template_arr_skel, new_lines, seed_image_locs, cap_filled_num, cap_total_num)
    
    # create "good" region in which to count pixels to measure line width
    cap_image_bounds = []
    cap_ratios[i] = []
    for angle in cap_measure_angs[i]:
        cc = np.cos(angle)
        ss = np.sin(angle)
        temp = []
        for p1 in cap_measure_bounds:
            p1 = geo.rotate_point(p1, angle, scaling/microns_per_pixel_x,
                                  scaling/microns_per_pixel_y, (x_loc, y_loc))
            temp.append(p1)
        cap_bounds = np.array(temp)
        row, col = draw.polygon(cap_bounds[:,1], cap_bounds[:,0])
        cap_img_num = np.sum(binary[row, col])
        cap_thr_num = len(np.where(cap_rotated_template_arr[row, col]==1)[0])
        cap_ratios[i].append(cap_img_num / cap_thr_num)
        temp.append(temp[0])
        cap_image_bounds.append(np.array(temp))

    pf.capacitor_line_width_fig(binary, cap_rotated_template_arr, cap_image_bounds, cap_ratios)

    # accounting to save to pickle file
    cap_filled_legs[i] =   {'0 seeded': {'leg 0' : filled_nums[0][0]/total_nums[0][0],
                                         'leg 1' : filled_nums[1][0]/total_nums[1][0]},
                            '1 seeded': {'leg 0' : filled_nums[1][1]/total_nums[1][1],
                                         'leg 1' : filled_nums[0][1]/total_nums[0][1]}}
    cap_flagged_break[i] =  {'leg 0' : True if cap_filled_legs[i]['0 seeded']['leg 0'] < 0.999 else False,
                             'leg 1' : True if cap_filled_legs[i]['1 seeded']['leg 1'] < 0.999 else False}
    cap_flagged_bridge[i] = {'leg 0' : True if cap_filled_legs[i]['0 seeded']['leg 1'] > 0.001 else False,
                             'leg 1' : True if cap_filled_legs[i]['1 seeded']['leg 0'] > 0.001 else False}
    cap_filled[i] = cap_filled_num / cap_total_num
    cap_flagged[i] = True if cap_filled[i] < 0.999 else False

# ------------------------
# SAVE DATA TO PICKLE FILE
# ------------------------

info_dict = {'center coord':           center_coord,
             'rotation angle':         final_rot_angle,
             'capacitor translations': cap_translations}
with open(args.save_dir + 'location_info_%s.pkl'%(str(args.index).zfill(3)), 'wb') as f:
    pk.dump(info_dict, f, protocol=pk.HIGHEST_PROTOCOL)

ind_filled = {0: filled_num_long / total_num_long,
              1: filled_num_broken / total_num_broken}
ind_flagged = {0: True if filled_num_long / total_num_long < 0.99 else False,
               1: True if filled_num_broken / total_num_broken < 0.99 else False}
overspill_fracs = [overspill_nums[i] / overspill_tots[i] for i in range(3)]
cap_overspill_fracs = {0: {i: cap_overspill_num[0][i] / cap_overspill_tot[0][i] for i in range(2)},
                       1: {i: cap_overspill_num[1][i] / cap_overspill_tot[1][i] for i in range(2)}}

out_dict = {0: {'inductor': {'thickness ratio': ratios[0],
                             'image nums':      img_nums[0],
                             'template nums':   thr_nums[0],
                             'box nums':        box_nums[0],
                             'percent filled':  filled_num_long / total_num_long,
                             'flagged':         True if filled_num_long / total_num_long < 0.99 else False,
                             'overspill':       overspill_fracs,
                             'questionable':    True if np.any(np.array(overspill_fracs) > 0.05) else False},
                'capacitor':{'thickness ratio': cap_ratios[0],
                             'percent filled':  cap_filled_legs[0],
                             'flagged break':   cap_flagged_break[0],
                             'flagged bridge':  cap_flagged_bridge[0],
                             'overspill':       cap_overspill_fracs[0],
                             'questionable':    True if cap_overspill_fracs[0][0] > 0.05 or cap_overspill_fracs[0][1] > 0.05 else False}},
            1: {'inductor': {'thickness ratio': ratios[1],
                             'image nums':      img_nums[1],
                             'template nums':   thr_nums[1],
                             'box nums':        box_nums[1],
                             'percent filled':  filled_num_broken / total_num_broken,
                             'flagged':         True if filled_num_broken / total_num_broken < 0.99 else False,
                             'overspill':       overspill_fracs,
                             'questionable':    True if np.any(np.array(overspill_fracs) > 0.05) else False},
                'capacitor':{'thickness ratio': cap_ratios[1],
                             'percent filled':  cap_filled_legs[1],
                             'flagged break':   cap_flagged_break[1],
                             'flagged bridge':  cap_flagged_bridge[1],
                             'overspill':       cap_overspill_fracs[1],
                             'questionable':    True if cap_overspill_fracs[1][0] > 0.05 or cap_overspill_fracs[1][1] > 0.05 else False}}}


with open(args.save_dir + 'stats_%s.pkl'%(str(args.index).zfill(3)), 'wb') as f:
    pk.dump(out_dict, f, protocol=pk.HIGHEST_PROTOCOL)
