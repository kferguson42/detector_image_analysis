import numpy as np
import pickle as pk
# the following imports will break unless you have set your PYTHONPATH as stated in the README
from geometry_utils import construct_region_around_region
from mkid_analysis.make_capacitor_template import draw_capacitor_lines
from mkid_analysis.mkid_mapping_utils import image_num_to_capacitor_info
from mkid_analysis import analysis_params

def get_inductor_bounds(aux_dir='/home/kferguson/software/detector_image_analysis/spt4/', pad=3):
    ignore = {'left':[[4], [4040,4047], [8110]],
              'right':[[2], [4065,4072], [8108]],
              'vertical':[[3136,3143], [11630,11637]]
              } # we don't want to include the big boxes at the end of the inductor

    bounds = {}
    for i, obj in enumerate(['left', 'right', 'vertical']):
        edge_lines = pk.load(open(aux_dir + '%s_meander_physical_units.pkl'%(obj), 'rb'))
        ig = ignore[obj]

        if obj == 'vertical':
            edge_lines = edge_lines[0:ig[0][0]] + edge_lines[ig[0][1]:ig[1][0]] + edge_lines[ig[1][1]:]
        else:
            edge_lines = edge_lines[ig[0][0]:ig[1][0]] + edge_lines[ig[1][1]:ig[2][0]]

        min_x = np.min(np.asarray(edge_lines)[:,0][:,0])
        max_x = np.max(np.asarray(edge_lines)[:,0][:,0])
        min_y = np.min(np.asarray(edge_lines)[:,0][:,1])
        max_y = np.max(np.asarray(edge_lines)[:,0][:,1])
        bounds[obj] = [[min_x-pad, min_y-pad], [min_x-pad, max_y+pad],
                       [max_x+pad, max_y+pad], [max_x+pad, min_y-pad]]
    return bounds

def get_capacitor_bounds(index, rot_angle, image_mode='spt4_dense_chip', pad=3):
    start1, span1, _, start2, span2, _ = image_num_to_capacitor_info(index, image_mode, trim=True)
    starting_angles = [start1+np.pi, start2+np.pi]
    angle_spans = [span1, span2]

    params = analysis_params.cap_design_params
    line_width = params['line_width']
    line_spacing = params['line_spacing']
    num_tines = params['num_tines']

    vertices = []
    for i in range(2):
        outer_radius = params['outer_radius'] + params['radius_adjustments'][i]
        inner_radius = outer_radius - (num_tines*(line_width + line_spacing) - line_spacing)

        # draw arcs
        angs = np.linspace(np.pi/2 - starting_angles[i], np.pi/2 - (starting_angles[i] + angle_spans[i]), num=200)
        inner_arc = [inner_radius*np.cos(angs), inner_radius*np.sin(angs)]
        outer_arc = [outer_radius*np.cos(angs[::-1]), outer_radius*np.sin(angs[::-1])]
        pts = np.append(inner_arc, outer_arc, axis=1).swapaxes(0,1)
        vertices.append(np.array(construct_region_around_region(pts, pad)))

    return vertices
