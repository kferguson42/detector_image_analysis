import numpy as np
# for the following import to work, you must add the repository to your PYTHONPATH (see README)
from mkid_analysis import analysis_params

def image_num_to_position(image_num, mode):
    '''
    Takes a (0-indexed) number denoting which image in the stack
    this is and returns the location of the bolometer (in microns)
    on the wafer. Assumes images have been captured as programmed
    in the ZEN_microscope_imaging repository.
    '''
    pixs = []

    feed_width = 30
    cpw_gap = 25
    pix_radius = 1150
    circle_sep = feed_width+2*cpw_gap
    tile_sidelength = 2*pix_radius-circle_sep
    factor = np.sqrt(3)/2.#np.sin(60*np.pi/180.)
    design_offset_x = -13000
    design_offset_y = -11000
    offset = tile_sidelength/2.

    if mode == 'spt4_220v5_UC':
        dense_cols = [0, 1, 3, 4, 5, 7, 8]
        sparse_cols = [1, 4, 7]
        off = {0:-1, 1:1}

        for mini in range(4):
            row_polarity = 1
            x = tile_sidelength/2.
            y = tile_sidelength/2.

            for j in range(9):
                x = tile_sidelength/2.
                if row_polarity > 0:
                    x += offset
                y += factor*tile_sidelength
                row_polarity*= -1

                for i in range(9):
                    filled = False
                    x += tile_sidelength
                    if mini // 2 == 0:
                        if i in dense_cols:
                            filled = True
                    else:
                        if row_polarity == 1 and i in sparse_cols:
                            filled = True
                    if filled:
                        pixs.append((x + design_offset_x + (off[mini%2]*16000),
                                     y + design_offset_y - (off[mini//2]*16000)))

    elif mode == 'spt4_MUXv1_UC':
        cols = [1,4,7]
        off = {0:-1, 1:1}

        for mini in range(4): # loop over smaller chips
            row_polarity = 1
            x = tile_sidelength/2.
            y = tile_sidelength/2.

            for j in range(9): # loop over rows
                x = tile_sidelength/2.
                if row_polarity > 0:
                    x += offset
                y += factor*tile_sidelength
                row_polarity*= -1

                for i in range(9): # loop over columns
                    filled = False
                    x += tile_sidelength
                    if mini // 2 == 1:
                        if i in cols and j != 8:
                            filled = True
                    else:
                        if i in cols and row_polarity == 1:
                            filled = True
                    if filled:
                        pixs.append((x + design_offset_x + (off[mini%2]*16000),
                                     y + design_offset_y - (off[mini//2]*16000)))

    elif mode == 'spt4_sparse_chip':
        sparse_cols = [1,4,7]
        off = {0:-1, 1:1}

        row_polarity = 1
        x = tile_sidelength/2.
        y = tile_sidelength/2.

        for j in range(9):
            x = tile_sidelength/2.
            if row_polarity > 0:
                x += offset
            y += factor*tile_sidelength
            row_polarity*= -1

            for i in range(9):
                filled = False
                x += tile_sidelength
       	        if i in sparse_cols:
                    # alignment marks are on top and bottom so need to rotate x & y 90 degrees
                    pixs.append(((-1.*(y + design_offset_y)),
                                x + design_offset_x))
    elif mode == 'spt4_dense_chip':
        row_polarity = 1
        x = tile_sidelength/2.
        y = tile_sidelength/2.
        temp_pixs = []
        for j in range(9):
            x = tile_sidelength/2.
            if row_polarity > 0:
                x += offset
            y += factor*tile_sidelength
            row_polarity*= -1

            for i in range(9):
                x += tile_sidelength
                temp_pixs.append((x+design_offset_x, y+design_offset_y))
        # no real alignment marks, so use the centers of the
        # far left and far right pixels in the middle row.
        # translate array so the center pixel in this row is at (0, 0)
        for p in temp_pixs:
            x, y = temp_pixs[40]
            pixs.append((p[0]-x, p[1]-y))

    return pixs[image_num]

def image_num_to_capacitor_info(image_num, mode, trim=False):
    '''
    Returns the starting angle and angle span (in radians) of the x- and y-capacitors
    for a detector with a given image number
    '''
    params = analysis_params.cap_design_params
    line_width = params['line_width']
    line_spacing = params['line_spacing']
    rail_width = params['rail_width']
    outer_radius = params['outer_radius']
    num_tines = params['num_tines']

    inner_radius = outer_radius - (num_tines*(line_width + line_spacing) - line_spacing)
    
    if mode == 'spt4_sparse_chip':
        # arc length array copied from spt4_design repo: MUX/create_spt4_MUXv1_UC.py
        arc_lengths = [1894.0,896.0,1848.0,879.0,1804.0,863.0,1762.0,847.0,1721.0,832.0,1681.0,817.0,
                       1643.0,802.0,1606.0,788.0,1570.0,773.0,1535.0,760.0,1502.0,746.0,1469.0,733.0,
                       1437.0,721.0,1407.0,708.0,1377.0,696.0,1348.0,684.0,1320.0,673.0,1293.0,661.0,
                       1267.0,650.0,1241.0,640.0,1216.0,629.0,1192.0,619.0,1168.0,608.0,1146.0,599.0, 1, 1]

    elif mode == 'spt4_dense_chip':
        # arc length array copied from spt4_design repo: cpw_220v7/create_spt4_220v7_UCw016.py
        x_sizes = [100, 1800, 1500, 1200]
        y_sizes = [100, 900, 700, 500]
        x_trim = 6
        y_trim = 4
        indices = [1,2,3,1,2,3,1,2,3, 2,3,1,2,3,1,2,3,1, 1,2,3,1,2,3,1,2,3, 2,3,1,2,3,1,2,3,1, 1,2,3,1,2,3,1,2,3,
                   2,3,1,2,3,1,2,3,1, 1,2,3,1,2,3,1,2,3, 2,3,1,2,3,1,2,3,1, 1,2,3,1,2,3,1,2,3, 0,0]
        bank_counts = [0,0,0,0]

        arc_lengths = []
        trim_sizes = []
        for i in indices:
            arc_lengths.append(x_sizes[i%4])
            arc_lengths.append(y_sizes[i%4])
            trim_sizes.append(x_trim*(bank_counts[i]) + 18)
            trim_sizes.append(y_trim*(bank_counts[i]) + 18)
            bank_counts[i]+=1

    cap_size_x, cap_size_y = arc_lengths[image_num*2], arc_lengths[image_num*2+1]
    if trim:
        trim_size_x, trim_size_y = trim_sizes[image_num*2], trim_sizes[image_num*2+1]
        x_trim_angle = trim_size_x/(outer_radius - line_width - line_spacing/2.)
        y_trim_angle = trim_size_y/(outer_radius - line_width - line_spacing/2.)
    x_width = num_tines*(line_width + line_spacing) - line_spacing
    x_span = 2*cap_size_x/(2*outer_radius - x_width)
    y_width = num_tines*(line_width + line_spacing) - line_spacing
    y_span = 2*cap_size_y/(2*outer_radius - y_width)

    if mode == 'spt4_sparse_chip':
        x_start_angle = np.pi - x_span/2.
        y_start_angle = (2*np.pi) - y_span/2.
    elif mode == 'spt4_dense_chip':
        x_start_angle = 100. * (np.pi/180.)
        y_start_angle = -30. * (np.pi/180.)

    if trim:
        return x_start_angle, x_span, x_trim_angle, y_start_angle, y_span, y_trim_angle
    else:
        return x_start_angle, x_span, y_start_angle, y_span



