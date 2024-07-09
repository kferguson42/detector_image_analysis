'''
This file contains all of the plotting code for the main analyze_mkid_pixel.py file.
These functions are meant only to be used in that script, and as such they reference
explicit variables from that script. The functions are here, and not in that file,
just to make that one a bit cleaner and focused explicitly on the analysis steps.
'''

import matplotlib.pyplot as plt
import numpy as np
# for the following import to work, you must add the repository to your PYTHONPATH (see README)
import mkid_analysis.analysis_params as aparams
params = aparams.params_20x
line_frac = params['line_frac']
mask_frac = params['mask_frac']

def inductor_overspill_fig(plot_temp, overspill_nums, overspill_tots):
    plt.figure(figsize=(10,10))
    plt.imshow(plot_temp, cmap=plt.cm.gray, origin='lower')
    plt.title('filled pixels in circles: %d / %d = %.2f%%, %d / %d = %.2f%%, %d / %d = %.2f%%'%(
        overspill_nums[0], overspill_tots[0], (overspill_nums[0]/overspill_tots[0])*100.,
        overspill_nums[1], overspill_tots[1], (overspill_nums[1]/overspill_tots[1])*100.,
        overspill_nums[2], overspill_tots[2], (overspill_nums[2]/overspill_tots[2])*100.))
    plt.savefig(save_dir + 'inductor_overspill_circle_%s.png'%(str(index).zfill(3)), dpi=200)
    plt.close()

def template_subtraction_fig(image, binary, rotated_template_arr, ratios):
    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
    ax = axes.ravel()
    yl = y_loc-y//32
    yh = y_loc+y//32
    xl = x_loc-x//32
    xh = x_loc+x//32
    ax[0].imshow(image[yl:yh, xl:xh], cmap=plt.cm.gray, origin='lower')
    ax[0].set_title('Input image')
    ax[1].imshow(binary[yl:yh, xl:xh], cmap=plt.cm.gray, origin='lower')
    ax[1].set_title('Binary thresholded image')
    ax[2].imshow(rotated_template_arr[yl:yh, xl:xh], cmap=plt.cm.gray, origin='lower')
    ax[2].set_title('Template')
    ax[3].imshow((binary-rotated_template_arr)[yl:yh, xl:xh], cmap=plt.cm.gray, origin='lower')
    ax[3].set_title('Difference (binary - template),\n%.2f (%.2f) image/template pixels in long (broken) inductor'%(np.mean(ratios[0]), np.mean(ratios[1])))
    plt.tight_layout()
    plt.savefig(save_dir + 'template_subtraction_%s.png'%(str(index).zfill(3)), dpi=200)
    plt.close()

def main_inductor_fig(image, skel, filtered, filled, lines, fit_mask, km, seed_image_locs, cutoff_x, cutoff_y, maxs, ratio_long, ratio_broken):
    yl = cutoff_y; yh = -cutoff_y; xl = cutoff_x; xh = -cutoff_x
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    ax = axes.ravel()

    i = 0
    ax[i].imshow(image[yl:yh, xl:xh], cmap=plt.cm.gray, origin='lower')
    ax[i].set_title('Input image')

    i = 1
    ax[i].imshow(skel[yl:yh, xl:xh], cmap=plt.cm.gray, origin='lower')
    ax[i].set_title('Skeletonized Scharr edges')

    i = 2
    ax[i].imshow(image[yl:yh, xl:xh], cmap=plt.cm.gray, origin='lower')
    for num_line, line in enumerate(lines):
        p0, p1 = line
        p0 += np.array([x//2-x//line_frac, y//2-y//line_frac])
        p1 += np.array([x//2-x//line_frac, y//2-y//line_frac])
        if km.labels_[num_line]:
            ax[i].plot((p0[0]-xl, p1[0]-xl), (p0[1]-yl, p1[1]-yl), 'C1')
        else:
            ax[i].plot((p0[0]-xl, p1[0]-xl), (p0[1]-yl, p1[1]-yl), 'C2')
    ax[i].set_title('Input image + Hough lines')

    i = 3
    ax[i].imshow(filtered[yl:yh, xl:xh], origin='lower')
    ax[i].plot(x_loc-xl, y_loc-yl, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
    ax[i].set_title('Matched filtered skeleton (best orientation), max = %.3f'%(maxs))

    i = 4
    ax[i].imshow(image[yl:yh, xl:xh], cmap=plt.cm.gray, origin='lower')
    for l in fit_mask:
        p1, p2 = l
        if p1[0] >= x_loc-x//mask_frac and p1[0] < x_loc+x//mask_frac and p2[0] >= x_loc-x//mask_frac and p2[0] < x_loc+x//mask_frac and \
           p1[1] >= y_loc-y//mask_frac and p1[1] < y_loc+y//mask_frac and p2[1] >= y_loc-y//mask_frac and p2[1] < y_loc+y//mask_frac:
            ax[i].plot([p1[0]-xl, p2[0]-xl], [p1[1]-yl, p2[1]-yl], 'C1')
        else:
            ax[i].plot([p1[0]-xl, p2[0]-xl], [p1[1]-yl, p2[1]-yl], 'C2')
    ax[i].set_title('Template overlay')

    i = 5
    ax[i].imshow(filled, cmap=plt.cm.gray, origin='lower')
    for j in range(2):
        ax[i].plot(seed_image_locs[j][0], seed_image_locs[j][1], 'xC%s'%(j))
    ax[i].set_title('filled skeleton, %.2f%% (%.2f%%) filled pixels in long (broken) inductor'%(ratio_long*100, ratio_broken*100))

    for a in ax:
        a.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_dir + 'flood_fill_%s.png'%(str(index).zfill(3)), dpi=200)
    plt.close()

def capacitor_template_fit(half_filled, cap_rotated_template_arr_skel, seed_image_locs, filled_nums, total_nums, adjustment=False, cap_diff=None):
    fig, axes = plt.subplots(figsize=(10,5), nrows=1, ncols=2)
    axs = axes.ravel()
    for j in range(2):
        temp = np.zeros(half_filled[j].shape)
        temp[half_filled[j] == 1] = 1
        temp[cap_rotated_template_arr_skel == 1] = 0.5
        ax = axs[j]
        ax.imshow(temp, cmap=plt.cm.gray, origin='lower')
        p = seed_image_locs[j]
        ax.plot(p[0], p[1], 'rx')
        ax.set_xlim([p[0]-700, p[0]+700])
        ax.set_ylim([p[1]-700, p[1]+700])
        ax.set_title('%.2f%% seeded side filled, %.2f%% unseeded side filled'%(
            (filled_nums[0][j]/total_nums[0][j]) * 100, (filled_nums[1][j]/total_nums[1][j]) * 100))
    plt.suptitle('not moved' if not adjustment else 'moved by %d in y, %d in x'%(cap_diff[1], cap_diff[0]))
    fig.tight_layout()
    tag = 'with' if adjustment else 'no'
    fig.savefig(save_dir + 'capacitor_template_fit_%s_%d_%s_adjustment.png'%(str(index).zfill(3), i, tag), dpi=200)
    plt.close()

def capacitor_overspill_fig(temp, cap_overspill_num, cap_overspill_tot, j):
    plt.figure(figsize=(10,10))
    plt.imshow(temp, cmap=plt.cm.gray, origin='lower')
    plt.title('%d / %d = %.2f%% filled pixels in circle'%(cap_overspill_num[i][j], cap_overspill_tot[i][j],
                                                          (cap_overspill_num[i][j]/cap_overspill_tot[i][j])*100.))
    plt.savefig(save_dir + 'capacitor_overspill_circle_%s_%d_%d.png'%(str(index).zfill(3), i, j), dpi=200)
    plt.close()

def capacitor_xcorr_fig(xcorrs):
    plt.figure(figsize=(10,10))
    plt.imshow(xcorrs, origin='lower')
    plt.title('x-correlations around original center point')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_dir + 'capacitor_xcorrs_%s_%d.png'%(str(index).zfill(3), i), dpi=200)
    plt.close()

def capacitor_skeleton_fig(new_skel, new_lines, seed_image_locs):
    plt.figure(figsize=(10,10))
    min_x = np.min(np.array(new_lines)[:,0,0])
    max_x = np.max(np.array(new_lines)[:,0,0])
    min_y = np.min(np.array(new_lines)[:,0,1])
    max_y = np.max(np.array(new_lines)[:,0,1])
    
    plt.imshow(new_skel, cmap=plt.cm.gray, origin='lower')
    for p in seed_image_locs:
        plt.plot(p[0], p[1], 'rx')
    plt.xlim([min_x-100, max_x+100])
    plt.ylim([min_y-100, max_y+100])
    plt.title('seeds at (%d, %d) and (%d, %d)'%(seed_image_locs[0][0], seed_image_locs[0][1],
                                                seed_image_locs[1][0], seed_image_locs[1][1]))
    plt.tight_layout()
    plt.savefig(save_dir + 'capacitor_skeleton_%s_%d.png'%(str(index).zfill(3), i), dpi=200)
    plt.close()

def capacitor_flood_fill(filled, cap_rotated_template_arr_skel, new_lines, seed_image_locs, cap_filled_num, cap_total_num):
    plt.figure(figsize=(10,10))
    min_x = np.min(np.array(new_lines)[:,0,0])
    max_x = np.max(np.array(new_lines)[:,0,0])
    min_y = np.min(np.array(new_lines)[:,0,1])
    max_y = np.max(np.array(new_lines)[:,0,1])
    
    temp = np.zeros(filled.shape)
    temp[filled] = 1
    temp[cap_rotated_template_arr_skel == 1] = 0.5
    plt.imshow(temp, cmap=plt.cm.gray, origin='lower')
    for p in seed_image_locs:
        plt.plot(p[0], p[1], 'rx')
    plt.xlim([min_x-100, max_x+100])
    plt.ylim([min_y-100, max_y+100])
    plt.title('%.2f%% of skeleton filled'%((cap_filled_num / cap_total_num)*100))
    plt.tight_layout()
    plt.savefig(save_dir + 'capacitor_flood_fill_%s_%d.png'%(str(index).zfill(3), i), dpi=200)
    plt.close()

def capacitor_line_width_fig(binary, cap_rotated_template_arr, cap_image_bounds, cap_ratios):
    fig, axes = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True, figsize=(10,5))
    ax = axes.ravel()
    ax[0].imshow(binary, cmap=plt.cm.gray, origin='lower')
    ax[0].set_title('Binarized input image')
    ax[1].imshow(cap_rotated_template_arr, cmap=plt.cm.gray, origin='lower')
    for rect in cap_image_bounds:
        ax[0].plot(rect[:,0], rect[:,1], 'r')
        ax[1].plot(rect[:,0], rect[:,1], 'r')
    ax[1].set_title('Template')
    cap_image_bounds = np.array(cap_image_bounds)
    ax[1].set_xlim([np.min(cap_image_bounds[:,:,0])-25, np.max(cap_image_bounds[:,:,0])+25])
    ax[1].set_ylim([np.min(cap_image_bounds[:,:,1])-25, np.max(cap_image_bounds[:,:,1])+25])
    fig.suptitle('%.2f image/template pixels'%(np.mean(cap_ratios[i])))
    plt.tight_layout()
    plt.savefig(save_dir + 'capacitor_line_width_%s_%d.png'%(str(index).zfill(3), i), dpi=200)
    plt.close()
