import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
import os
import pickle5
import missingpy
import argparse as ap
from glob import glob
# the following import will break unless you have set your PYTHONPATH as stated in the README
import geometry_utils
'''
This script reads in the visual characteristics as saved by the
image_features.py script and writes them out + saves them in the
format expected by the ML scripts.
'''
p = ap.ArgumentParser()
p.add_argument('-w', '--wafer-num', type=int, default=148)
p.add_argument('--feature-dir', type=str)
p.add_argument('--image-dir', type=str)
p.add_argument('--save-dir', type=str)
args = p.parse_args()

wafer = args.wafer_num
feature_dir = args.feature_dir
image_dir = args.image_dir
save_dir = args.save_dir

feature_files = sorted(glob(feature_dir + 'bolo_feature*.pkl'))

out_arr = []
feature_list = []
tes_nums = []
groups = []

# populate output array
# each row contains features for single TES
for i, fname in enumerate(feature_files):
    feats = pk.load(open(fname, 'rb'))
    stuff = fname.split('/')[-1].split('_')
    rot_group = int(stuff[2][-1])
    for j, tes_num in enumerate(feats.keys()):
        this_arr = []
        tes_nums.append(tes_num)
        groups.append(rot_group)

        for feat_name, feat in feats[tes_num].items():
            if 'area' in feat_name:
                this_arr.append(feat['area'])
                if i == 0 and j == 0:
                    feature_list.append(feat_name)
            elif 'vertices' in feat_name:
                mean_x, mean_y = np.mean(feat, axis=0)
                # need to swap penultimate and final points to make valid shapely polygon
                mean_x[[-2, -1]] = mean_x[[-1, -2]]
                mean_y[[-2, -1]] = mean_y[[-1, -2]]
                if np.any(np.isnan(mean_x)) or np.any(np.isnan(mean_y)):
                    this_arr.append(np.nan)
                else:
                    perim = geometry_utils.perimeter_around_points(mean_x, mean_y)
                    this_arr.append(perim)
                if i == 0 and j == 0:
                    feature_list.append(feat_name.split(' ')[0] + ' perimeter')
            elif 'ness' in feat_name: #'roughness' and 'edge weirdness'
                for k, v in feat.items():
                    this_arr.append(v)
                    if i == 0 and j == 0:
                        feature_list.append(feat_name+' '+k)
            elif 'template' in feat_name:
                this_arr.append(np.mean(feat))
                if np.mean(feat) < 0.05:
                    print(tes_num, rot_group, len(feat))
                if i == 0 and j == 0:
                    feature_list.append(feat_name)
            elif '(deg)' in feat_name: #'angle (deg)', orientation of bolo
                this_arr.append(np.mean(feat))
                if i == 0 and j == 0:
                    feature_list.append(feat_name.split(' ')[0])

        # add another feature for whether any other features for this bolo are nans
        if i == 0 and j == 0:
            feature_list.append('has nan')
        if np.any(np.isnan(np.array(this_arr))):
            this_arr.append(1)
        else:
            this_arr.append(0)

        if len(this_arr) != 14:
            print(tes_num, rot_group)
        out_arr.append(this_arr)

# apply feature scaling
# ACTUALLY, DON'T. feaure scaling not necessary for tree-based models
#out_arr = np.array(out_arr)
#m = np.mean(out_arr, axis=0)
#s = np.std(out_arr, axis=0)
#out_arr = (out_arr - m) / s

out_arr = np.array(out_arr)
groups = np.array(groups)

for i in range(len(feature_list)):
    if sum(np.isnan(out_arr[:,i])) > 0:
        print(i)
        for j in np.where(np.isnan(out_arr[:,i]))[0]:
            print(tes_nums[j], groups[j])

# histogram each feature
print(out_arr.shape)
print(len(feature_list))
print(len(tes_nums))
print(len(feature_files))
print(groups.shape)

out_dict = {'feature names':feature_list,
            'features':out_arr,
            'bolos':tes_nums}
pk.dump(out_dict, open(save_dir + 'w%s_input_features.pkl'%(wafer), 'wb'), protocol=pk.HIGHEST_PROTOCOL)

for i in range(len(feature_list)):
    plt.figure()
    x = out_arr[:,i]
    _, bins = np.histogram(x[~np.isnan(x)], bins=21)

    for k in range(6):
        plt.hist(out_arr[:,i][np.where(groups == k)], bins=bins, color='C%d'%(k), label='rotated%d'%(k), alpha=0.5)

    t = feature_list[i]
    plt.title(t)
    if 'area' in t:
        plt.xlabel('square pixels')
    elif 'perimeter' in t:
        plt.xlabel('pixels')
    elif 'angle' in t:
        plt.xlabel('deg')
    else:
        plt.xlabel('arbitrary units')
    plt.legend()
    plt.savefig(save_dir + 'feature_hist_%s.png'%(t.replace(' ', '_')))
    plt.close()

