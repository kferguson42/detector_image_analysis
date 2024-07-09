import numpy as np
import matplotlib.pyplot as plt
# for the following import to work, you must add the repository to your PYTHONPATH (see README)
from mkid_analysis import analysis_params

def draw_capacitor_lines(starting_angle, angle_span, invert_dir = False, outer_radius=925.0,
                         trim_angle=None):
    '''
    Creates the mask for a capacitor that subtends the given angle.
    '''

    params = analysis_params.cap_design_params
    line_width = params['line_width']
    line_spacing = params['line_spacing']
    rail_width = params['rail_width']
    rail_extend_in = params['rail_extend_in']
    rail_extend_out = params['rail_extend_out']
    num_tines = params['num_tines']

    inner_radius = outer_radius - (num_tines*(line_width + line_spacing) - line_spacing)

    do_trim = trim_angle is not None
    
    # draw
    arcs_in = []
    arcs_out = []
    end_lines = []
    connecting_lines = []

    # arc starts at +y and goes in clockwise-direction
    arc = lambda radius, ang: (radius*np.cos(np.pi/2 - ang), radius*np.sin(np.pi/2 - ang))
    for i in range(16):
        skip = False
        if (i == 15 and do_trim) or (invert_dir and i == 14 and do_trim):
            span = angle_span - trim_angle
        else:
            span = angle_span

        for j in range(2):
            these_pts = []
            if invert_dir:
                if i == 15:
                    skip = True
                    break
                radius = inner_radius + 2*i*(line_spacing + line_width) + (j+2)*line_width
            else:
                radius = inner_radius + 2*i*(line_spacing + line_width) + j*line_width
            end_gap_angle = line_spacing / radius
            #end_gap_angle = 0

            if invert_dir:
                angs = np.linspace(starting_angle + angle_span, starting_angle + end_gap_angle + (angle_span-span), 200)
            else:
                angs = np.linspace(starting_angle, starting_angle + span - end_gap_angle, 200)

            for ang in angs:
                these_pts.append(arc(radius, ang))
            if j == 0:
                arcs_in.append(these_pts)
            else:
                arcs_out.append(these_pts)

        if skip:
            continue
        # draw connecting bits
        #if invert_dir:
        #    print(i, len(arcs_in), skip)
        #    l = [(arcs_in[i][0][0], arcs_out[i][0][0]), (arcs_in[i][0][1], arcs_out[i][0][1])]
        #else:
        l = [(arcs_in[i][-1][0], arcs_out[i][-1][0]), (arcs_in[i][-1][1], arcs_out[i][-1][1])]
        end_lines.append(l)
        if i > 0:
            l = [(arcs_out[i-1][0][0], arcs_in[i][0][0]), (arcs_out[i-1][0][1], arcs_in[i][0][1])]
            connecting_lines.append(l)

    # draw rectangle
    rectangle = []
    if invert_dir:
        dx1i = -1. * (rail_extend_in + line_width + line_spacing) * np.sin(angle_span + starting_angle)
        dy1i = (rail_extend_in + line_width + line_spacing) * np.cos(angle_span + starting_angle)
        dx1o = -1. * (rail_extend_out + line_width + line_spacing) * np.sin(angle_span + starting_angle)
        dy1o = (rail_extend_out + line_width + line_spacing) * np.cos(angle_span + starting_angle)
        dx2 = rail_width * np.cos(angle_span + starting_angle)
        dy2 = -1. * rail_width * np.sin(angle_span + starting_angle)
    else:
        dx1i = -1. * rail_extend_in * np.sin(starting_angle)
        dy1i = rail_extend_in * np.cos(starting_angle)
        dx1o = -1. * rail_extend_out * np.sin(starting_angle)
        dy1o = rail_extend_out * np.cos(starting_angle)
        dx2 = -1. *rail_width * np.cos(starting_angle)
        dy2 = rail_width * np.sin(starting_angle)

    inner1 = (arcs_in[0][0][0] + dx1i, arcs_in[0][0][1] - dy1i)
    outer1 = (arcs_out[-1][0][0] - dx1o, arcs_out[-1][0][1] + dy1o)
    inner2 = (inner1[0] + dx2, inner1[1] + dy2)
    outer2 = (outer1[0] + dx2, outer1[1] + dy2)
    rectangle.append([(arcs_out[-1][0][0], outer1[0]), (arcs_out[-1][0][1], outer1[1])])
    rectangle.append([(outer1[0], outer2[0]), (outer1[1], outer2[1])])
    rectangle.append([(outer2[0], inner2[0]), (outer2[1], inner2[1])])
    rectangle.append([(inner2[0], inner1[0]), (inner2[1], inner1[1])])
    rectangle.append([(inner1[0], arcs_in[0][0][0]), (inner1[1], arcs_in[0][0][1])])

    # get all lines in order (is this actually necessary?)
    all_lines = []
    all_lines.extend(rectangle)
    for i in range(len(arcs_in)):
        for j, p1 in enumerate(arcs_in[i]):
            if j != len(arcs_in[i])-1:
                p2 = arcs_in[i][j+1]
                all_lines.append([(p1[0], p2[0]), (p1[1], p2[1])])
        all_lines.append(end_lines[i])
        for j, p1 in enumerate(arcs_out[i][::-1]):
            if j != len(arcs_out[i])-1:
                p2 = arcs_out[i][::-1][j+1]
                all_lines.append([(p1[0], p2[0]), (p1[1], p2[1])])
        if i < len(arcs_in) - 1:
            all_lines.append(connecting_lines[i])

    '''
    #fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    colors = plt.cm.jet(np.linspace(0, 1, len(all_lines)))
    for i, l in enumerate(all_lines):
        ax.plot(l[0], l[1], color=colors[i])
    #ax.plot([400,800], [400,800], 'C0')

    #ax.set_aspect('equal')
    #plt.legend()
    #plt.show()
    '''

    return all_lines


def trace_capacitor_gap(starting_angle, angle_span, outer_radius=925.0):
    '''
    d
    '''
    params = analysis_params.cap_design_params
    line_width = params['line_width']
    line_spacing = params['line_spacing']
    rail_width = params['rail_width']
    rail_extend_in = params['rail_extend_in']
    rail_extend_out = params['rail_extend_out']
    num_tines = params['num_tines']

    inner_radius = outer_radius - (num_tines*(line_width + line_spacing) - line_spacing)

    arc_lines = []
    connecting_lines = []
    arc = lambda radius, ang: (radius*np.cos(np.pi/2 - ang), radius*np.sin(np.pi/2 - ang))

    for i in range(num_tines - 1):
        radius = inner_radius + line_width + (line_spacing/2) + i*(line_spacing + line_width)
        end_gap_angle = line_spacing / radius
        these_lines = []
        angs = np.linspace(starting_angle + (end_gap_angle/2), starting_angle + angle_span - (end_gap_angle/2), 200)

        for ang in angs:
            these_lines.append(arc(radius, ang))
        arc_lines.append(these_lines)

        if i > 0:
            if i % 2:
                l = [(arc_lines[i-1][0][0], arc_lines[i][0][0]), (arc_lines[i-1][0][1], arc_lines[i][0][1])]
            else:
                l = [(arc_lines[i-1][-1][0], arc_lines[i][-1][0]), (arc_lines[i-1][-1][1], arc_lines[i][-1][1])]
            connecting_lines.append(l)

    all_lines = []
    for i in range(num_tines - 1):
        if i % 2:
            for j, p1 in enumerate(arc_lines[i]):
                if j != len(arc_lines[i])-1:
                    p2 = arc_lines[i][j+1]
                    all_lines.append([(p1[0], p2[0]), (p1[1], p2[1])])
        else:
            for j, p1 in enumerate(arc_lines[i][::-1]):
                if j != len(arc_lines[i])-1:
                    p2 = arc_lines[i][::-1][j+1]
                    all_lines.append([(p1[0], p2[0]), (p1[1], p2[1])])
        if i < len(arc_lines) - 1:
            all_lines.append(connecting_lines[i])

    return all_lines


if __name__ == '__main__':
    fig = plt.figure(figsize=(10,10))
    lines1 = draw_capacitor_lines(0.2, np.pi/4, invert_dir=False)
    lines1.extend(draw_capacitor_lines(0.2, np.pi/4, invert_dir=True))
    lines2 = trace_capacitor_gap(0.2, np.pi/4)

    ax = fig.gca()
    #c = np.zeros(len(lines1))
    #c[:int(len(lines1)/2)] = np.linspace(0, 1, int(len(lines1)/2))
    #c[int(len(lines1)/2):] = np.linspace(0, 1, int(len(lines1)/2))
    c = np.linspace(0, 1, len(lines1))
    colors = plt.cm.jet(c)


    for i, l in enumerate(lines1):
        ax.plot(l[0], l[1], color='C0')#colors[i])
    for i, l in enumerate(lines2):
        ax.plot(l[0], l[1], color='C1')


    #ax.plot([400,800], [400,800], 'C0')
    ax.set_aspect('equal')
    #plt.legend()
    #plt.show()
    '''
    arr = np.zeros((1000,1000))
    from skimage import draw, segmentation

    diff = np.array([25,25])
    for l in lines1:
        p1 = np.array([l[0][0], l[1][0]]) + diff
        p2 = np.array([l[0][1], l[1][1]]) + diff
        #rr, cc, val = draw.line_aa(int(np.round(p1[1])), int(np.round(p1[0])),
        #                           int(np.round(p2[1])), int(np.round(p2[0])))
        rr, cc = draw.line(int(np.round(p1[1])), int(np.round(p1[0])),
                           int(np.round(p2[1])), int(np.round(p2[0])))
        arr[rr, cc] = 1

    fill_pts = [(653, 650),
                (638, 650)
            ]

    fig = plt.figure(figsize=(10,10))
    for p in fill_pts:
        plt.plot(p[0], p[1], 'rx')

    x = segmentation.flood_fill(arr, fill_pts[0][::-1], True, connectivity=1, in_place=False)
    arr[np.logical_and(x == 1, arr != 1)] = 0.5
    x = segmentation.flood_fill(arr, fill_pts[1][::-1], True, connectivity=1, in_place=False)
    arr[np.logical_and(x == 1, arr != 1)] = 0.5
    plt.imshow(arr, cmap=plt.cm.gray, origin='lower')
    '''
    plt.show()

