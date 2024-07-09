import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import itertools
from skimage import filters, transform, io, morphology, color, feature, draw
from sklearn.cluster import KMeans
from copy import deepcopy
from glob import glob
from scipy.ndimage import gaussian_filter
import argparse as ap
# the following import will break unless you have set your PYTHONPATH as stated in the README
import geometry_utils

'''
This script reads in an image of an SPT-3G TES bolometer island, determines
the location of the TES in the image, and calculates a set of potentially
interesting features including:
- area + perimeter (as well as area + perimeter of smaller regions not
  overlapped by other parts of bolometer architecture)
- surface roughness
- edge roughness
- correlation between image edges and fabrication template
'''

p = ap.ArgumentParser()
p.add_argument('-i', '--image', type=str, default='/mnt/c/Users/fergu/Downloads/TES_images/',
               help="Can be either a single file or a directory.")
p.add_argument('-s', '--save-dir', type=str, default='/mnt/c/Users/fergu/Downloads/TES_images/results/')
p.add_argument('-m', '--mask-file', type=str, default='/home/kferguson/spt3g_wafer_ML/wafer_mask_lines_updated.pkl')
p.add_argument('-t', '--threshold', type=float, default=0.2,
               help="Normalized value above which to set Meijering edges to 1 (and below to 0) before skeletonization")
p.add_argument('-n', '--num-iterations', type=int, default=100)
p.add_argument('-w', '--wafer-num', type=int, default=0)
args = p.parse_args()

if args.image.endswith('/'):
    images = sorted(glob(args.image + 'image_*'))
    fout = 'bolo_features_%s.pkl'%(args.image.split('/')[-1])
else:
    images = [args.image]
    rotation_group = args.image.split('/')[-2].split('_')[3]
    fout = 'bolo_features_%s_TES_%s.pkl'%(rotation_group, int(args.image.split('/')[-1].split('.')[0].split('_')[-1]))

save_dir = args.save_dir
mask_file = args.mask_file
mask_lines = pk.load(open(mask_file, 'rb'))

bolo_angles = {}
areas = {}
corners = {}
bling_corners = {}
lead_corners = {}
reduced_areas = {}
more_reduced_areas = {}
edge_detection_maxima = {}
surface_roughness = {}
match_filter = {}
for im in images:
    tes_num = int(im.split('/')[-1].split('_')[-1].split('.')[0])
    rotation_group = im.split('/')[-2].split('_')[3]
    areas[tes_num] = []
    corners[tes_num] = []
    bolo_angles[tes_num] = []
    bling_corners[tes_num] = []
    lead_corners[tes_num] = []
    reduced_areas[tes_num] = []
    more_reduced_areas[tes_num] = []
    match_filter[tes_num] = []
    f = io.imread(im)
    y,x,_ = f.shape

    image = color.rgb2gray(f)
    imshape = (image.shape[1], image.shape[0])

    edges = filters.meijering(image)
    edges /= np.max(edges)

    thresh = args.threshold
    e = deepcopy(edges)
    e[e<thresh] = 0
    e[e>=thresh] = 1
    skeleton = morphology.skeletonize(e)

    # first find general bolometer island location in image
    lines = transform.probabilistic_hough_line(skeleton, threshold=10, line_length=20, line_gap=10)
    ang, _ = geometry_utils.hough_lines_to_angle(lines)
    s = {'regular':np.sin(ang), 'flipped':np.sin(ang+np.pi)}
    c = {'regular':np.cos(ang), 'flipped':np.cos(ang+np.pi)}
    templates = {'regular':np.zeros(image.shape, dtype=np.uint8), 'flipped':np.zeros(image.shape, dtype=np.uint8)}
    fit_mask = {'regular':[], 'flipped':[]}
    for l in mask_lines:
        for o in ['regular', 'flipped']:
            p1 = np.array([(l[0][0]*c[o] - l[0][1]*s[o])+(imshape[0]/2), \
                           (l[0][0]*s[o] + l[0][1]*c[o])+(imshape[1]/2)])
            p2 = np.array([(l[1][0]*c[o] - l[1][1]*s[o])+(imshape[0]/2), \
                           (l[1][0]*s[o] + l[1][1]*c[o])+(imshape[1]/2)])
            row, col, val = draw.line_aa(int(np.round(p1[1])), int(np.round(p1[0])),
                                     int(np.round(p2[1])), int(np.round(p2[0])))
            templates[o][row, col] = val * 255
            fit_mask[o].append([p1, p2])

    filtered = {}
    for o in ['regular', 'flipped']:
        filtered[o] = feature.match_template(skeleton, templates[o], pad_input=True)
        filtered[o] = gaussian_filter(filtered[o], 1.5)

    if np.max(filtered['regular']) > np.max(filtered['flipped']):
        orientation = 'regular'
    else:
        orientation = 'flipped'

    fit_loc = np.unravel_index(np.argmax(filtered[orientation]), filtered[orientation].shape)
    x_loc, y_loc = fit_loc[::-1]

    # cut out smaller image centered on bolo and then run the actual line finding
    if y_loc < y//4:
        ybot = 0; ytop = y//2
    elif y_loc >= y//4 and y_loc < 3*(y//4):
        ybot = y_loc - y//4; ytop = y_loc + y//4
    else:
        ybot = y//2; ytop = y
    if x_loc < x//4:
        xbot = 0; xtop = x//2
    elif x_loc >= x//4 and x_loc < 3*(x//4):
        xbot = x_loc - x//4; xtop = x_loc + x//4
    else:
        xbot = x//2; xtop = x

    image = image[ybot:ytop, xbot:xtop]
    imshape = (image.shape[1], image.shape[0])

    edges = filters.meijering(image)
    edges /= np.max(edges)

    thresh = args.threshold
    e = deepcopy(edges)
    e[e<thresh] = 0
    e[e>=thresh] = 1
    skeleton = morphology.skeletonize(e)

    # probabilistic Hough algorithm utilizes random numbers, get better estimation for
    # area by calculating it a bunch of times
    for it in range(args.num_iterations):
        lines = transform.probabilistic_hough_line(skeleton, threshold=10, line_length=20, line_gap=10)

        ang, km = geometry_utils.hough_lines_to_angle(lines)
        len0 = len(km.labels_) - sum(km.labels_)
        len1 = sum(km.labels_)
        if len1 > len0:
            group = 1
            small = 0
        else:
            group = 0
            small = 1
        
        s = {'regular':np.sin(ang), 'flipped':np.sin(ang+np.pi)}
        c = {'regular':np.cos(ang), 'flipped':np.cos(ang+np.pi)}
        templates = {'regular':np.zeros(image.shape, dtype=np.uint8), 'flipped':np.zeros(image.shape, dtype=np.uint8)}
        fit_mask = {'regular':[], 'flipped':[]}
        for l in mask_lines:
            for o in ['regular', 'flipped']:
                p1 = np.array([(l[0][0]*c[o] - l[0][1]*s[o])+(imshape[0]/2), \
                               (l[0][0]*s[o] + l[0][1]*c[o])+(imshape[1]/2)])
                p2 = np.array([(l[1][0]*c[o] - l[1][1]*s[o])+(imshape[0]/2), \
                               (l[1][0]*s[o] + l[1][1]*c[o])+(imshape[1]/2)])
                row, col, val = draw.line_aa(int(np.round(p1[1])), int(np.round(p1[0])),
                                         int(np.round(p2[1])), int(np.round(p2[0])))
                templates[o][row, col] = val * 255
                fit_mask[o].append([p1, p2])

        filtered = {}
        for o in ['regular', 'flipped']:
            filtered[o] = feature.match_template(skeleton, templates[o], pad_input=True)
            filtered[o] = gaussian_filter(filtered[o], 1.5)
        
        if np.max(filtered['regular']) > np.max(filtered['flipped']):
            orientation = 'regular'
            match_filter[tes_num].append(np.max(filtered['regular']))
        else:
            orientation = 'flipped'
            match_filter[tes_num].append(np.max(filtered['flipped']))

        fit_loc = np.unravel_index(np.argmax(filtered[orientation]), filtered[orientation].shape)
        x_loc, y_loc = fit_loc[::-1]
        diff = np.array([fit_loc[1] - ((imshape[0])/2), fit_loc[0] - ((imshape[1])/2)])
        fit_mask[orientation] += diff

        tes_lines = fit_mask[orientation][156:160]
        tes_inds = []
        found_lines = [0,0,0,0]

        # find Hough lines that are best fit to expected TES position
        for i, tl in enumerate(tes_lines):
            if i in [1, 3]:
                g = group
            else:
                g = small
            max_ind = None
            max_len = 0

            verts = geometry_utils.construct_region_around_line(tl, 15)
            for j, l in enumerate(lines):
                # check if line is right orientation
                if km.labels_[j] != g:
                    continue

                if geometry_utils.is_point_in_region(l[0], verts) and \
                   geometry_utils.is_point_in_region(l[1], verts):
                    line_length = np.linalg.norm(np.array(l[1]) - np.array(l[0]))

                    if line_length > max_len:
                        max_ind = j
                        max_len = line_length

            if max_ind is not None:
                tes_inds.append(max_ind)
                found_lines[i] = 1
            else:
                continue
                
        good_lines = []
        for i, l_ind in enumerate(tes_inds):
            l = lines[l_ind]
            good_lines.append(l)

        xints = []; yints = []
        if len(tes_inds) == 4:
            for i, pair in enumerate(itertools.combinations(good_lines, r=2)):
                if i == 1 or i == 4:
                    continue
                xint, yint = geometry_utils.intersection_of_two_lines(pair[0], pair[1])
                xints.append(xint); yints.append(yint)

            areas[tes_num].append(geometry_utils.area_btwn_points(xints, yints))
            corners[tes_num].append([xints, yints])
            bolo_angles[tes_num].append(ang*(180./np.pi))
        else:
            areas[tes_num].append(np.nan)
            corners[tes_num].append([np.nan*np.ones(4), np.nan*np.ones(4)])
            bolo_angles[tes_num].append(np.nan)

        # first two are the bling lines, second two are actualy the lead lines
        bling_mask_lines = [fit_mask[orientation][232], fit_mask[orientation][251], fit_mask[orientation][136], fit_mask[orientation][142]]
        bling_inds = []
        bling_hough_lines = []
        vertss = []

        # find Hough lines that are best fit to expected position
        for i, tl in enumerate(bling_mask_lines):
            max_ind = None
            max_len = 0

            verts = geometry_utils.construct_region_around_line(tl, 15)
            vertss.append(verts)
            for j, l in enumerate(lines):
                # check if line is right orientation
                if km.labels_[j] != small:
                    continue

                if geometry_utils.is_point_in_region(l[0], verts) and \
                   geometry_utils.is_point_in_region(l[1], verts):
                    line_length = np.linalg.norm(np.array(l[1]) - np.array(l[0]))

                    if line_length > max_len:
                        max_ind = j
                        max_len = line_length

            if max_ind is not None:
                bling_inds.append(max_ind)
                bling_hough_lines.append(lines[max_ind])
            else:
                continue

        xints_b = []; yints_b = []; small_lines = []
        if len(bling_inds) == 4 and found_lines[1] == 1 and found_lines[3] == 1:
            small_lines = deepcopy(bling_hough_lines)
            if found_lines[2] == 1:
                small_lines.extend([good_lines[-3], good_lines[-1]])
            else:
                small_lines.extend(good_lines[-2:])

            for i, pair in enumerate(itertools.combinations(small_lines, r=2)):
                # dark magic to find correct indices
                if i in [0,1,2,5,6,9,14]:
                    continue
                xint, yint = geometry_utils.intersection_of_two_lines(pair[0], pair[1])
                xints_b.append(xint); yints_b.append(yint)

            reduced_areas[tes_num].append(geometry_utils.area_btwn_points(xints_b[-4:], yints_b[-4:]))
            more_reduced_areas[tes_num].append(geometry_utils.area_btwn_points(xints_b[:4], yints_b[:4]))
            bling_corners[tes_num].append([xints_b[:4], yints_b[:4]])
            lead_corners[tes_num].append([xints_b[-4:], yints_b[-4:]])
        else:
             reduced_areas[tes_num].append(np.nan)
             more_reduced_areas[tes_num].append(np.nan)
             bling_corners[tes_num].append([np.nan*np.ones(4), np.nan*np.ones(4)])
             lead_corners[tes_num].append([np.nan*np.ones(4), np.nan*np.ones(4)])

        if it == 0:
            # plot found lines on top of original image
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(image, cmap=plt.cm.gray, origin='lower')
            ax[0].set_title('Input image + Hough lines, angle %.1f deg'%(ang*(180./np.pi)))

            ax[1].imshow(edges, cmap=plt.cm.gray, origin='lower')
            ax[1].plot(imshape[0]/2, imshape[1]/2, 'xC2')
            ax[1].set_title('Meijering filtered image')

            ax[2].imshow(skeleton, cmap=plt.cm.gray, origin='lower')
            ax[2].set_title('Skeletonized meijering edges')
            
            ax[3].imshow(image, cmap=plt.cm.gray, origin='lower')
            for num_line, line in enumerate(lines):
                p0, p1 = line
                if km.labels_[num_line]:
                    ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]), 'C1')
                else:
                    ax[3].plot((p0[0], p1[0]), (p0[1], p1[1]), 'C2')
            ax[3].set_title('Input image + Hough lines, angle %.1f deg'%(ang*(180./np.pi)))

            ax[4].imshow(filtered[orientation], origin='lower')
            ax[4].plot(x_loc, y_loc, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
            ax[4].set_title('matched filtered skeleton (better orientation), max = %.3f'%(match_filter[tes_num][-1]))

            ax[5].imshow(image, cmap=plt.cm.gray, origin='lower')
            for i, l in enumerate(bling_hough_lines):
                p0, p1 = l
                ax[5].plot((p0[0], p1[0]), (p0[1], p1[1]), 'C6')
            for i, l in enumerate(good_lines):
                p0, p1 = l
                ax[5].axline(p0, xy2=p1, color='k')
                ax[5].plot((p0[0], p1[0]), (p0[1], p1[1]), 'C%d'%(i))
                if len(good_lines) == 4:
                    ax[5].plot(xints[i], yints[i], 'rx')
            if len(small_lines) == 6:
                for i in range(8):
                    ax[5].plot(xints_b[i], yints_b[i], 'C%d.'%((i//4) + 1))
            ax[5].set_title('Best guess at TES lines, bling lines')

            for a in ax:
                a.set_axis_off()

            plt.tight_layout()
            plt.savefig(save_dir + 'TES_lines_%s_%s.png'%(rotation_group, tes_num))
            plt.close()
            print(tes_num)

    # calculate edge thickness with mean vertex location
    mean_verts_tes = []
    mean_verts_bling = []
    for it in range(len(corners[tes_num])):
        xs, ys = corners[tes_num][it]
        temp = []
        for vert in zip(xs, ys):
            temp.append(vert)
        mean_verts_tes.append(temp)
    mean_verts_tes = np.nanmean(mean_verts_tes, axis=0)
    for it in range(len(bling_corners[tes_num])):
        xs, ys = bling_corners[tes_num][it]
        temp = []
        for vert in zip(xs, ys):
            temp.append(vert)
        mean_verts_bling.append(temp)
    mean_verts_bling = np.nanmean(mean_verts_bling, axis=0)

    if np.any(np.isnan(mean_verts_bling)) or np.any(np.isnan(mean_verts_tes)):
        surface_roughness[tes_num] = {'mean': np.nan, 'std':np.nan}
        edge_detection_maxima[tes_num] = {'num pixels':np.nan, 'mean':np.nan, 'std':np.nan}
        continue
    
    # get points in right order
    mean_verts_tes[[-2, -1]] = mean_verts_tes[[-1, -2]]
    mean_verts_bling[[-2, -1]] = mean_verts_bling[[-1, -2]]

    maxs = feature.peak_local_max(edges, indices=False, threshold_rel=thresh)
    verts, max_y, min_y, max_x, min_x = geometry_utils.construct_region_around_region(mean_verts_tes, 5, return_extrema=True)
    vb = geometry_utils.construct_region_around_region(mean_verts_bling, -10)

    maxs[:min_y, :] = 0
    maxs[max_y:, :] = 0
    maxs[min_y:max_y, :min_x] = 0
    maxs[min_y:max_y, max_x:] = 0
    tes_brightness = []
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            if not geometry_utils.is_point_in_region((i,j), verts):
                maxs[j, i] = 0
            if geometry_utils.is_point_in_region((i,j), vb):
                tes_brightness.append(image[j,i])

    surface_roughness[tes_num] = {'mean':np.mean(tes_brightness), 'std':np.std(tes_brightness)}
    edge_detection_maxima[tes_num] = {'num pixels':len(maxs[maxs != 0]), 'mean':np.mean(edges[maxs != 0]), 'std':np.std(edges[maxs != 0])}

for k in areas.keys():
    area_arr = areas[k]
    m = np.nanmean(area_arr)
    s = np.nanstd(area_arr)
    areas[k] = {'area':m, 'err':s / np.sqrt(len(area_arr))}
for k in reduced_areas.keys():
    area_arr = reduced_areas[k]
    m = np.nanmean(area_arr)
    s = np.nanstd(area_arr)
    reduced_areas[k] = {'area':m, 'err':s / np.sqrt(len(area_arr))}
    area_arr = more_reduced_areas[k]
    m = np.nanmean(area_arr)
    s = np.nanstd(area_arr)
    more_reduced_areas[k] = {'area':m, 'err':s / np.sqrt(len(area_arr))}

all_features = {}
for k in areas.keys():
    all_features[k] = {'full area': areas[k],
                       'half reduced area': reduced_areas[k],
                       'full reduced area': more_reduced_areas[k],
                       'TES vertices': corners[k],
                       'lead vertices': lead_corners[k],
                       'bling vertices': bling_corners[k],
                       'roughness': surface_roughness[k],
                       'edge weirdness': edge_detection_maxima[k],
                       'template correlation': match_filter[k],
                       'angle (deg)': bolo_angles[k]
        }

with open(save_dir + fout, 'wb') as f:
    pk.dump(all_features, f, protocol=pk.HIGHEST_PROTOCOL)
