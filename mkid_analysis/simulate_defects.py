import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from skimage import filters, transform, io, morphology, color, feature, draw, segmentation
from skimage.measure import find_contours
from copy import deepcopy
from glob import glob
import argparse as ap
# for the following imports to work, you must add the repository to your PYTHONPATH (see README)
import geometry_utils
from mkid_analysis import object_boundaries
from mkid_analysis.make_capacitor_template import draw_capacitor_lines
from mkid_analysis.analysis_params import params_20x
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

p = ap.ArgumentParser()
p.add_argument('-i', '--image', type=str)
p.add_argument('--info', type=str, help="location of translation/rotation info file")
p.add_argument('-m', '--mode', type=str, choices=['same', 'opposite'], default='opposite')
p.add_argument('-l', '--location', type=str, choices=['inductor', 'ind', 'capacitor', 'cap', 'both'], default='both')
p.add_argument('-s', '--save-dir', type=str, default='/home/kferguson/Analyses/DetectorAnalysis/spt4_detector_files/results/updated_stitching/')
p.add_argument('--aux-dir', type=str, default='/home/kferguson/software/spt3g_wafer_ML/spt4/')
p.add_argument('--index', type=int,
               help="Detector indexing number (used to determine capacitor location)")
p.add_argument('-n', '--num-defects', type=int)
p.add_argument('-d', '--defect-size', type=int)
p.add_argument('--seed', type=int, default=-1,
               help="RNG seed to use. If not specified, will default to the pixel index.")
p.add_argument('--suffix', type=str, default='',
               help="suffix to append to saved files")
args = p.parse_args()

if args.seed == -1:
    np.random.seed(args.index)
else:
    np.random.seed(args.seed)

suffix = '_%s'%(args.suffix) if args.suffix != '' else ''

def add_defect(shape, size, location, input_image, mode, cutout_pad, cap_or_ind, blend=False):
    def blend_boundaries(shape, cutout_pad, blend_pad = 3):
        overlap = np.zeros((2*(size+cutout_pad), 2*(size+cutout_pad)))
        temp = np.ones((2*(size+cutout_pad), 2*(size+cutout_pad)))
        if shape == 'square':
            row_per, col_per = draw.rectangle_perimeter((cutout_pad, cutout_pad), extent=2*size, shape=(2*(size+cutout_pad), 2*(size+cutout_pad)))
        elif shape == 'circle':
            row_per, col_per = draw.circle_perimeter(size+cutout_pad, size+cutout_pad, size+1, shape=(2*(size+cutout_pad), 2*(size+cutout_pad)))

        overlap[row_per, col_per] = 1
        temp[row_per, col_per] = 0
        morphology.flood_fill(temp, (2*size+cutout_pad+2, size+cutout_pad), 0, connectivity=1, in_place=True)
        for i in range(blend_pad-1):
            bounds = segmentation.find_boundaries(temp, connectivity=1, mode='inner')
            overlap[bounds] = i+2
            temp[bounds] = 0
        morphology.flood_fill(overlap, (size+cutout_pad, size+cutout_pad), blend_pad+1, connectivity=1, in_place=True)
        return overlap

    assert shape in ['square', 'circle']
    assert mode in ['same', 'opposite']

    rr, cc = location
    if mode == 'same':
        val = on_val if binary[rr,cc] == 1 else off_val
        defect_type = 'sputter' if binary[rr,cc] == 1 else 'etch'
    elif mode == 'opposite':
        val = off_val if binary[rr,cc] == 1 else on_val
        defect_type = 'etch' if binary[rr,cc] == 1 else 'sputter'

    if shape == 'square':
        if not blend:
            rows, cols = draw.rectangle((rr-size, cc-size), extent=2*size, shape=input_image.shape)
        else:
            overlap = blend_boundaries('square', cutout_pad)

    elif shape == 'circle':
        if not blend:
            rows, cols = draw.disk((rr, cc), size, shape=input_image.shape)
        else:
            overlap = blend_boundaries('circle', cutout_pad)

    if not blend:
        input_image[rows, cols] = val
    else:
        contours = find_contours(input_image[rr-(size+cutout_pad):rr+(size+cutout_pad), cc-(size+cutout_pad):cc+(size+cutout_pad)], np.mean([off_val, on_val]))
        orig = input_image[rr-(size+cutout_pad):rr+(size+cutout_pad), cc-(size+cutout_pad):cc+(size+cutout_pad)]
        input_image[rr-(size+cutout_pad):rr+(size+cutout_pad), cc-(size+cutout_pad):cc+(size+cutout_pad)] = 0.25*overlap*val + 0.25*(4-overlap)*orig
        #binary_defect = np.array(overlap, dtype=bool)
        binary_defect = np.zeros(overlap.shape)
        binary_defect[overlap >= 2] = 1
        defect_contour = find_contours(binary_defect, 0.5)[0]
        broken_contours = 0
        contour_mins = []
        for c in contours:
            this_min = 2*cutout_pad + 1
            for p in c:
                if geometry_utils.is_point_in_region(p[::-1], defect_contour):
                    broken_contours += 1
                    this_min = 2*cutout_pad + 1
                    break
                else:
                    dists = []
                    pt = p[::-1]
                    for dp in defect_contour:
                        d_pt = dp[::-1]
                        dists.append(np.sqrt((pt[0] - d_pt[0])**2 + (pt[1] - d_pt[1])**2))
                    md = np.min(dists)
                    if md < this_min:
                        this_min = md
            contour_mins.append(this_min)
        if mode == 'same' and broken_contours == 1 and cap_or_ind in ['capacitor', 'cap'] and len(contours) > 1:
            total_min = contour_mins[np.argsort(contour_mins)[1]]
        else:
            total_min = np.min(contour_mins)
        return broken_contours, defect_type, defect_contour, total_min


save_dir = args.save_dir
mask_file = args.aux_dir + 'pixel_mask_physical_units.pkl'
mask_lines = pk.load(open(mask_file, 'rb'))

params = params_20x
microns_per_pixel_x = params['microns_per_pixel_x']
microns_per_pixel_y = params['microns_per_pixel_y']

f = io.imread(args.image)
y,x,_ = f.shape
image = color.rgb2gray(f)
xx = x/2
yy = y/2

info = pk.load(open(args.info, 'rb'))
center = np.array(info['center coord']) + 100 #need to add 100 because we shave off some edge pixels in the analysis script
rot_angle = info['rotation angle']
cap_diff = info['capacitor translations']
c = np.cos(rot_angle)
s = np.sin(rot_angle)

valid = np.zeros(image.shape)
boundary_pad = 1
bounds = {}
if args.location in ['inductor', 'ind', 'both']:
    ind_bounds_physical_units = object_boundaries.get_inductor_bounds(aux_dir=args.aux_dir, pad=boundary_pad)
    for obj in ind_bounds_physical_units.keys():
        bounds[obj] = []
        for p in ind_bounds_physical_units[obj]:
            bounds[obj].append(np.array([
                (p[0]*c - p[1]*s)*(1./microns_per_pixel_x)+center[0], \
                (p[0]*s + p[1]*c)*(1./microns_per_pixel_y)+center[1]]))
        bounds[obj] = np.array(bounds[obj])
        row, col = draw.polygon(bounds[obj][:,1], bounds[obj][:,0], image.shape)
        valid[row,col] = 1
if args.location in ['capacitor', 'cap', 'both']:
    cap_bounds_physical_units = object_boundaries.get_capacitor_bounds(args.index, rot_angle, image_mode='spt4_dense_chip', pad=boundary_pad)
    for i in range(2):
        obj = 'capacitor%d'%(i)
        bounds[obj] = []
        for p in cap_bounds_physical_units[i]:
            bounds[obj].append(np.array([
                (p[0]*c - p[1]*s)*(1./microns_per_pixel_x)+center[0]+cap_diff[i][0], \
                (p[0]*s + p[1]*c)*(1./microns_per_pixel_y)+center[1]+cap_diff[i][1]]))
        bounds[obj] = np.array(bounds[obj])
        row, col = draw.polygon(bounds[obj][:,1], bounds[obj][:,0], image.shape)
        valid[row,col] = 1

thresh_image = filters.threshold_otsu(image[100:-100, 100:-100][y//2-4000:y//2+4000, x//2-4000:x//2+4000])
binary = image > thresh_image
ind_num = np.sum(binary[valid==1])
tot_num = len(np.where(valid==1)[0])

on_val =  np.median(image[np.logical_and(valid==1, binary==1)])
off_val = np.median(image[np.logical_and(valid==1, binary==0)])

rows, cols = np.where(valid==1)
fig, axes = plt.subplots(ncols=args.num_defects, nrows=2, figsize=(4*args.num_defects, 8))
ax = axes.ravel()
d = args.defect_size
pad = 10 if args.location in ['capacitor', 'cap'] else 100-args.defect_size
out_dict = {'defect types':[],
            'broken contours':[],
            'min dist':[],
            'full break':[]
            }

for i in range(args.num_defects):
    ind = np.random.randint(0,len(row))
    rr = rows[ind]
    cc = cols[ind]

    contours = find_contours(image[rr-100:rr+100, cc-100:cc+100], np.mean([off_val, on_val]))
    bc, defect_type, defect_contour, min_dist = add_defect('circle', d, (rr,cc), image, args.mode, pad, args.location, blend=True)
    out_dict['defect types'] .append(defect_type)
    out_dict['broken contours'].append(bc)
    out_dict['min dist'].append(min_dist)
    if args.location == 'capacitor':
        if args.mode == 'same':
            out_dict['full break'].append(True if bc >= 3 else False)
        elif args.mode == 'opposite':
            out_dict['full break'].append(True if bc >= 2 else False)
    elif args.location == 'inductor':
        if defect_type == 'etch':
            out_dict['full break'].append(True if bc >= 2 else False)
        elif defect_type == 'sputter':
            # These types of defects can't be found by the algorithm as-is
            out_dict['full break'].append(False)

    ax[i].imshow(image[rr-100:rr+100, cc-100:cc+100], cmap=plt.cm.gray, origin='lower')
    for c in contours:
        ax[i].plot(c[:,1], c[:,0])
    ax[i].plot(defect_contour[:,1]+(100-pad)-d, defect_contour[:,0]+(100-pad)-d, 'b')
    ax[i].set_title('%d broken contours, %.2f pixels\n to closest non-broken contour'%(bc, min_dist))
    ax[i+args.num_defects].imshow(image[rr-100:rr+100, cc-100:cc+100], cmap=plt.cm.gray, origin='lower')

with open(args.save_dir + 'defect_stats/stats_%s_%s_%d_defects_size_%d_%s_style%s.pkl'%(str(args.index).zfill(3), args.location, args.num_defects, d, args.mode, suffix), 'wb') as f:
    pk.dump(out_dict, f, protocol=pk.HIGHEST_PROTOCOL)

plt.savefig(save_dir + 'defect_images/image_%s_%s_%d_defects_size_%d_%s_style%s.png'%(str(args.index).zfill(3), args.location, args.num_defects, d, args.mode, suffix))
plt.close()
io.imsave(save_dir + 'image_%s_%s_%d_defects_size_%d_%s_style%s.png'%(str(args.index).zfill(3), args.location, args.num_defects, d, args.mode, suffix), image)



