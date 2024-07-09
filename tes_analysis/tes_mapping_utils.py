import numpy as np

rotated_rows = [-7, -6, -3, -2, 1, 2, 5, 6, 9]

def image_num_to_physical_bolo_name(num, return_row_num=False):
    '''
    Takes a (0-indexed) number denoting which image in the stack
    this is and returns the physical bolometer name on the wafer.
    Assumes images have been captured as programmed in the
    ZEN_microscope_imaging repository.
    '''
    rows = {9:np.arange(1,11,1,dtype=int),
            8:np.arange(11,22,1,dtype=int),
            7:np.arange(22,34,1,dtype=int),
            6:np.arange(34,47,1,dtype=int),
            5:np.arange(47,61,1,dtype=int),
            4:np.arange(61,76,1,dtype=int),
            3:np.arange(76,92,1,dtype=int),
            2:np.arange(92,109,1,dtype=int),
            1:np.arange(109,127,1,dtype=int),
            0:np.arange(127,146,1,dtype=int),
            -1:np.arange(146,164,1,dtype=int),
            -2:np.arange(164,181,1,dtype=int),
            -3:np.arange(181,197,1,dtype=int),
            -4:np.arange(197,212,1,dtype=int),
            -5:np.arange(212,226,1,dtype=int),
            -6:np.arange(226,239,1,dtype=int),
            -7:np.arange(239,251,1,dtype=int),
            -8:np.arange(251,262,1,dtype=int),
            -9:np.arange(262,272,1,dtype=int)}
    row_lengths = {9:10, 8:11, 7:12, 6:13, 5:14, 4:15, 3:16,
                   2:17, 1:18, 0:19, -1:18, -2:17, -3:16, -4:15,
                   -5:14, -6:13, -7:12, -8:11, -9:10}

    # when imaging a "right-side up" wafer, we start in the bottom left
    # (pixel 262), go along the row (i.e. to pixel 271), then step up
    # a row (i.e. to pixel 251)

    num = int(num)
    bolo_num = num % 6
    pix_ind = (num // 6) + 1 #pixel numbers start at 1

    row_num = -9
    row_ind = 0
    full_ind = 0
    for i in range(1,272):
        row_ind += 1
        full_ind += 1
        if full_ind == pix_ind:
            pix_num = rows[row_num][row_ind-1]
            break

        if row_ind == row_lengths[row_num]:
            row_num += 1
            row_ind = 0

    pixel_layout = {0:[150, 'x'], 1:[150, 'y'],
                    2:[90, 'y'],  3:[220, 'y'],
                    4:[90, 'x'],  5:[220, 'x']}

    physical_name = '%d.%d.%s'%(pix_num, pixel_layout[bolo_num][0], pixel_layout[bolo_num][1])

    if return_row_num:
        return physical_name, row_num
    else:
        return physical_name

def image_num_to_position(image_num):
    '''
    Takes a (0-indexed) number denoting which image in the stack
    this is and returns the location of the bolometer (in microns)
    on the wafer. Assumes images have been captured as programmed
    in the ZEN_microscope_imaging repository.
    '''
    pixs = []

    N = 9 #10 pixels per hexagon side
    x_space = 6789.0  #in microns
    y_space = 5879.4  #in microns
    bolo_locs = [(0, 2107.5), (-2107.5,0),
                 (2102.5, 0), (2520.5, 0),
                 (0,-2102.5), (0,-2520.5)]

    #rotated_rows = [-9, -6, -5, -2, -1, 2, 3, 6, 7]
    rotated_rows = [-7, -6, -3, -2, 1, 2, 5, 6, 9]
    rotated_locs = []
    rotation = np.cos(np.pi/4.) # sin(pi/4) = cos(pi/4) so just use this
    for loc in bolo_locs:
        rotated_locs.append((loc[0]*rotation - loc[1]*rotation,
                             loc[0]*rotation + loc[1]*rotation))

    for n in range(-N,N+1):
        x_start = -(N-abs(n/2.))*x_space
        y_start = n*y_space
        for nn in range(0,(2*N+1)-abs(n)):
            x_pos = x_start + nn*x_space

            if n in rotated_rows:
                for loc in rotated_locs:
                    pixs.append((x_pos+loc[0], y_start+loc[1]))
            else:
                for loc in bolo_locs:
                    pixs.append((x_pos+loc[0], y_start+loc[1]))
    
    return pixs[image_num]

def image_num_to_readout_bolo_name(image_num, wafer_hwm_file):
    '''
    Takes a (0-indexed) number denoting which image in the stack
    this is and returns the readout name of the bolometer. Returns
    None if there is no readout name for the given bolometer in the
    hardware map. Assumes images have been captured as programmed
    in the ZEN_microscope_imaging repository.
    '''
    map_info = np.loadtxt(wafer_hwm_file, skiprows=1, dtype=str, delimiter='\t')

    physical_names = map_info[:,0]
    readout_names = map_info[:,2]

    phys_name = image_num_to_physical_bolo_name(image_num)
    ind = np.where(physical_names == phys_name)
    try:
        read = readout_names[ind[0][0]]
        return read
    except:
        #print('Couldn\'t find readout name for TES num %d, bolo %s.'%(image_num, phys_name))
        return None
