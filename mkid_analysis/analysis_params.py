from numpy import linspace, pi

params_20x = {
    'line_frac' : 24,
    'hough_thresh' : 5,
    'min_length' : 35,
    'gap' : 5,
    'angs' : linspace(-pi/2., pi/2., 900),
    'mask_frac' : 50,
    'scaling' : 1.0,
    'seed_locs' : [(45,8), (-109,0), (111,0)],
    'cutout_size' : 6000,
    'microns_per_pixel_x' : 600/2628, #0.228310502283105
    'microns_per_pixel_y' : 600/2628, #397.667/1752 #~1752px - 397.667um~
    #'questionable_thresh' : 140000,
    }

params_50x = {
    'line_frac' : 8,
    'hough_thresh' : 10,
    'min_length' : 80,
    'gap' : 5,
    'angs' : linspace(-pi/2., pi/2., 360),
    'mask_frac' : 25,
    'scaling' : 0.9986363636363635,
    'seed_locs' : [(110,20), (-272,0), (277,0)],
    'microns_per_pixel_x' : 0.091,
    'microns_per_pixel_y' : 0.091,
    #'questionable_thresh' : None, #haven't tested with 50x, probably won't use this mag
    }

ind_design_params = {
    'line_width' : 2,        # microns
    'leg_length' : 16,       # microns
    'leg_sep' : 2,           # microns
    'subst_height' : 500,    # microns
    'line_height': .03,      # microns
    'total_length': 8140,    # microns
    'n_turns_long' : 250,
    'n_turns_short': 240,
    }

cap_design_params = {
    'line_width' : 4.0,      # microns
    'line_spacing' : 4.0,    # microns
    'rail_width' : 6.0,      # microns
    'outer_radius' : 925.0,  # microns
    'rail_extend_in' : 20,
    'rail_extend_out' : 5,
    'num_tines' : 31,
    'radius_adjustments' : {0:1.48, 1:1.04},  # microns
    }
