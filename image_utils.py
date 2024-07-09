import numpy as np

def xcorr(template, image, search_region_center, search_width_pix):
    # Implementation of Eqn. (2) from http://www.scribblethink.org/Work/nvisionInterface/nip.pdf
    tmp_h, tmp_w = template.shape
    img_h, img_w = image.shape
    tmp_min = np.min(template); tmp_max = np.max(template)
    img_min = np.min(image); img_max = np.max(image)

    # Normalize pixel values
    # XXX: THIS PROBABLY WON'T WORK WITH THE DIFFERENT DATA TYPES BEING USED RIGHT NOW, NEED TO FIX THAT
    # Actually, seems to be fine; might be worth doublechecking though
    template = (template-tmp_min) / (tmp_max-tmp_min)
    template -= np.mean(template)
    if image.dtype != bool:
        image = (image-img_min) / (img_max-img_min)
    else:
        image = np.array(image, dtype=float)

    pad_length = search_width_pix//2
    min_row = pad_length + search_region_center[0] - (search_width_pix//2 + tmp_h//2)
    max_row = pad_length + search_region_center[0] + tmp_h//2 + tmp_h%2 - search_width_pix//2 + (search_width_pix-1)
    min_col = pad_length + search_region_center[1] - (search_width_pix//2 + tmp_w//2)
    max_col = pad_length + search_region_center[1] + tmp_w//2 + tmp_w%2 - search_width_pix//2 + (search_width_pix-1)
    bounds = np.array([min_row, tmp_h-max_row, min_col, tmp_w-max_col])

    if np.any(bounds < 0):
        pad_length += np.abs(np.min(bounds))
    image = np.pad(image, pad_length)
    corr_matrix = np.zeros((search_width_pix, search_width_pix))

    for i in range(0, search_width_pix):
        for j in range(0, search_width_pix):
            img_slice = image[i+pad_length+search_region_center[0]-(search_width_pix//2+tmp_h//2) : i+pad_length+search_region_center[0]+tmp_h//2-search_width_pix//2+tmp_h%2,
                              j+pad_length+search_region_center[1]-(search_width_pix//2+tmp_w//2) : j+pad_length+search_region_center[1]+(tmp_w//2)-(search_width_pix//2)+(tmp_w%2)]
            m = np.mean(img_slice)
            img_slice -= m
            corr = np.sum(template*img_slice)
            norm = np.sqrt(np.sum(template**2)) * np.sqrt(np.sum(img_slice**2))
            img_slice += m
            corr_matrix[i, j] = corr / norm
    return corr_matrix

def xcorr_same_size(template, image):
    # Implementation of Eqn. (2) from http://www.scribblethink.org/Work/nvisionInterface/nip.pdf
    tmp_h, tmp_w = template.shape
    img_h, img_w = image.shape
    tmp_min = np.min(template); tmp_max = np.max(template)
    img_min = np.min(image); img_max = np.max(image)

    # Normalize pixel values
    # XXX: THIS PROBABLY WON'T WORK WITH THE DIFFERENT DATA TYPES BEING USED RIGHT NOW, NEED TO FIX THAT
    # Actually, seems to be fine; might be worth doublechecking though
    template = (template-tmp_min) / (tmp_max-tmp_min)
    template -= np.mean(template)
    if image.dtype != bool:
        image = (image-img_min) / (img_max-img_min)
    else:
        image = np.array(image, dtype=float)

    m = np.mean(image)
    image -= m
    corr = np.sum(template*image)
    norm = np.sqrt(np.sum(template**2)) * np.sqrt(np.sum(image**2))
    image += m
    return corr / norm

def trim_template(template):
    def biggest_int_not_in_list(arr):
       limit = np.max(arr)+1
       for num in arr[::-1]:
           if num == limit-1:
               limit -= 1
           elif num < limit-1:
               return limit-1
           else:
               raise ValueError
    def smallest_int_not_in_list(arr):
       limit = -1
       for num in arr:
           if num == limit+1:
               limit += 1
           elif num > limit+1:
               return limit+1
           else:
               raise ValueError

    no_info_rows = np.argwhere(np.all(template==0, axis=1))
    no_info_cols = np.argwhere(np.all(template==0, axis=0))
    yl = smallest_int_not_in_list(no_info_rows)
    yh = biggest_int_not_in_list(no_info_rows)
    xl = smallest_int_not_in_list(no_info_cols)
    xh = biggest_int_not_in_list(no_info_cols)
    return template[yl:yh+1, xl:xh+1]
