import matplotlib
matplotlib.use('Agg')
from os import path
import numpy as np
import pandas as pd
import argparse as ap
import pickle as pk
from glob import glob
from skimage import io, color, filters, morphology
import m2stitch
import matplotlib.pyplot as plt

p = ap.ArgumentParser()
p.add_argument('-i', '--image-dir', type=str)
p.add_argument('-o', '--output', type=str, default='stitched_image.png')
p.add_argument('-e', '--edges', action='store_true', default=False,
               help="Whether to perform an edge detection filter on the "
               "input images before stitching.")
p.add_argument('-it', '--iterations', type=int, default=100000)
args = p.parse_args()

if args.image_dir.endswith('/'):
    imfiles = sorted(glob(args.image_dir + '*.png'))
else:
    imfiles = sorted(glob(args.image_dir + '/*.png'))
if len(imfiles) == 0:
    if args.image_dir.endswith('/'):
        imfiles = sorted(glob(args.image_dir + '*.jpg'))
    else:
        imfiles = sorted(glob(args.image_dir + '/*.jpg'))

#print(imfiles)
rows_pix = []
cols_pix = []
og_images = []
images = []

if len(imfiles) == 0:
    raise ValueError("No input files found!")

for im in imfiles:
    x = io.imread(im)
    og_images.append(x)
    if args.edges:
        temp = filters.gaussian(filters.scharr(color.rgb2gray(x)), sigma=1)
        images.append(temp)
    else:
        images.append(color.rgb2gray(x))
    stuff = im.split('/')[-1].split('_')[-1].split('-')
    rows_pix.append(int(stuff[1].split('y')[-1]))
    cols_pix.append(int(stuff[0].split('x')[-1]))
print(rows_pix)
print(cols_pix)
images = np.asarray(images)

rows = np.array(rows_pix)
cols = np.array(cols_pix)
rm = np.min(rows[rows != 0])
cm = np.min(cols[cols != 0])
rows = np.rint(rows/rm)
cols = np.rint(cols/cm)
rows = np.array(rows, dtype=int)
cols = np.array(cols, dtype=int)

print(images.shape)
# must be 3-dim, with each dimension meaning (tile_index,y,x)
print(rows)
# the row (second-last dim.) indices for each tile index. for example, [1,1,2,2,2,...]
print(cols)
# the column (last dim.) indices for each tile index. for example, [2,3,1,2,3,...]

position_initial_guess = np.array([rows_pix, cols_pix]).T
result_df, _, nccs = m2stitch.stitch_images(images, rows, cols, position_initial_guess=position_initial_guess,
                                      ncc_threshold=0.03, row_col_transpose=False, npeaks=100, peak_separation=10,
                                      full_output=True, iterations=args.iterations)
pk.dump(nccs, open(args.output.replace('.png', '_nccs.pkl'), 'wb'), protocol=pk.HIGHEST_PROTOCOL)

# do stitching
result_df["y_pos2"] = result_df["y_pos"] - result_df["y_pos"].min()
result_df["x_pos2"] = result_df["x_pos"] - result_df["x_pos"].min()

size_y = images.shape[1]
size_x = images.shape[2]

stitched_image_size = (
    result_df["y_pos2"].max() + size_y,
    result_df["x_pos2"].max() + size_x,
    3
)
stitched_image = np.zeros_like(og_images, shape=stitched_image_size)
for i, row in result_df.iterrows():
    stitched_image[
        row["y_pos2"] : row["y_pos2"] + size_y,
        row["x_pos2"] : row["x_pos2"] + size_x,
    ] = og_images[i]
total_rows = i

'''
# draw lines across image borders
for j, row in result_df.iterrows():
    val = int(j*(255/i))
    stitched_image[row["y_pos2"] : row["y_pos2"] + size_y, row["x_pos2"]] = (255-val,0,val)
    stitched_image[row["y_pos2"], row["x_pos2"] : row["x_pos2"] + size_x] = (255-val,0,val)
'''

pk.dump(result_df, open(args.output.replace('.png', '_dataframe.pkl'), 'wb'), protocol=pk.HIGHEST_PROTOCOL)
io.imsave(args.output, stitched_image)
