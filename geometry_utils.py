import numpy as np
import shapely.geometry as geo
from itertools import combinations
from copy import deepcopy

def rotate_point(pt, angle, scaling_x, scaling_y, offset):
    '''
    Rotates a point around the origin, scales the location (to convert between physical
    and pixel coordinates), and translates the result to a non-origin-centered location.
    
    Input args:
    - pt (array-like): The point to rotate, in form (x, y)
    - angle (float): The angle, in radians, to rotate by
    - scaling_x, scaling_y (float): The value by which the rotated point coordinates are
      multiplied (can be different for x and y)
    - offset (array-like): The final value to translate by, in form (x, y)
    
    Returns:
    - pt_out (numpy array): The rotated, scaled, and translated point, in form (x, y)
    '''
    c = np.cos(angle)
    s = np.sin(angle)
    pt_out = np.array([(pt[0]*c - pt[1]*s) * scaling_x + offset[0], \
                       (pt[0]*s + pt[1]*c) * scaling_y + offset[1]])
    return pt_out

def intersection_of_two_lines(line1, line2):
    '''
    Given two line segments, calculates the intersection
    point of the infinte lines passing through the two line
    segments.
    
    From https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

    Input args:
    - line1, line2 (array-like): contain the line endpoints in form [(xA, yA), (xB, yB)]

    Returns:
    - intersection (tuple): the point where the two lines intersect, in form (x, y)
    '''
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)
    
def area_btwn_points(x, y):
    '''
    Calculates the area inside of a polygon with an arbitrary number of sides.

    From https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates,
    with some additional sorting of points to make the algorithm work.

    Input args:
    - x, y (array-like): the x (or y) coordinates of the polygon vertices. must be the same size.

    Returns:
    - area (float): the area of the polygon
    '''
    x = np.array(x)
    y = np.array(y)
    x -= np.mean(x)
    y -= np.mean(y)
    #x -= np.min(x)
    #y -= np.min(y)
    s = np.argsort(np.arctan2(y, x))
    x = x[s]; y = y[s]
    correction = x[-1] * y[0] - y[-1]* x[0]
    main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
    area = 0.5*np.abs(main_area + correction)
    return area

def perimeter_around_points(x, y):
    '''
    Calculates the perimeter of a polygon with an arbitrary number of sides. Basically
    just a wrapper around the shapely function that does the same thing.

    Input args:
    - x, y (array-like): the x (or y) coordinates of the polygon vertices. must be the same size.

    Returns:
    - perimeter (float): the perimeter of the polygon
    '''
    vertices = []
    for pt in zip(x, y):
        vertices.append(pt)
    gon = geo.Polygon(vertices)
    return gon.length

def distance_btwn_point_and_line(point, line):
    '''
    Gives the shortest distance between a point and the line defined by a line segment.

    From https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points,
    though I switched around the points a bit to make more sense to me.

    Input args:
    - point (array-like): the point to use, in form (x, y)
    - line (array-like): endpoints of the line segment, in form [(xA, yA), (xB, yB)]

    Returns:
    - dist (float): the shortest distance between the point and the line
    '''
    p0, p1 = line
    num = np.abs(np.cross(p0-point, p1-point))
    den = np.linalg.norm(p0 - p1)
    return num/den

def lengthen_line_segment(line, increase):
    '''
    Given a line segment, increases its length symmetrically about its cener by a
    specified amount.

    Input args:
    - line (array-like): endpoints of the line segment, in form [(xA, yA), (xB, yB)]
    - increase (float): the amount by which to extend the length

    Returns:
    - new_line (list): the endpoint coordinates of the new line segment, in form
      [(xA, yA), (xB, yB)]
    '''
    p1, p2 = line
    delta_x = (p2[0] - p1[0]) / np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    delta_y = (p2[1] - p1[1]) / np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    new_p1 = (p1[0] - (increase/2)*delta_x, p1[1] - (increase/2)*delta_y)
    new_p2 = (p2[0] + (increase/2)*delta_x, p2[1] + (increase/2)*delta_y)
    return [new_p1, new_p2]

def construct_region_around_line(line, pad):
    '''
    given a line, return the vertices of a region around the line

    Input args:
    - line (array-like): endpoints of the line segment, in form [(xA, yA), (xB, yB)]
    - pad (int): the number of pixels to pad around the line

    Returns:
    - region (list): a list of points, each an (x,y) tuple, defining the vertices
    of the boundary region
    '''
    p0, p1 = line
    x0, y0 = p0
    x1, y1 = p1
    amp = np.sqrt(2)*pad
    theta = np.arctan2(y1-y0, x1-x0)
    beta = (np.pi/4.) - theta
    
    v0 = (x0 - amp*np.cos(beta), y0 + amp*np.sin(beta))
    v1 = (x1 + amp*np.sin(beta), y1 + amp*np.cos(beta))
    v2 = (x1 + amp*np.cos(beta), y1 - amp*np.sin(beta))
    v3 = (x0 - amp*np.sin(beta), y0 - amp*np.cos(beta))

    return [v0, v1, v2, v3]

def construct_region_around_region(pts, pad, return_extrema=False):
    '''
    given a region, like the one returned by construct_region_around_line, return
    the vertices of a larger region around the original region

    Input args:
    - pts (array-like): a list/tuple/whatever of (x, y) points defining the vertices
    of the region
    - pad (int): the number of pixels to pad around the region
    - return_extrema (bool, optional): if true, also returns the minimum and maximum
    x and y values of the output region

    Returns:
    - region (list): a list of points, each an (x,y) tuple, defining the vertices
    of the boundary region
    '''
    gon = geo.Polygon(pts)
    x, y = gon.buffer(pad).exterior.coords.xy
    verts = []
    for i in range(len(x)):
        verts.append((x[i], y[i]))
    if not return_extrema:
        return verts
    else:
        return verts, int(np.ceil(np.max(y))), int(np.floor(np.min(y))), int(np.ceil(np.max(x))), int(np.floor(np.min(x)))

def is_point_in_region(point, vertices):
    '''
    boolean check if a point is in a given region.

    Input args:
    - point (array-like): the point to use, in form (x, y)
    - vertices (array-like): a list/tuple/whatever of (x, y) points defining the vertices
    of the region
    '''
    p = geo.Point(point)
    gon = geo.Polygon(vertices)
    return gon.contains(p)

def hough_lines_to_angle(lines):
    '''
    Given a set of Hough lines (in practice, line segments), use K-means to cluster the lines
    into two groups by angle; then return the mean angle of the larger group.

    Input args:
    - lines (array-like): a set of lines, each in the format [(xA, yA), (xB, yB)]

    Returns:
    - ang (float): mean angle of the larger clustered group of lines
    - km (sklearn.cluster.KMeans): the K-means object, can be used later to get the group
    for a chosen line and things like that
    '''
    from sklearn.cluster import KMeans
    angles = []
    unit_circle_positions = []
    for line in lines:
        p0, p1 = line
        # +y direction is down (dumb), so flip y-sign
        # Due to the probabilistic_hough_line convention
        # of the order of points returned, this makes all
        # angles in the range [0, np.pi]
        p0 = (p0[0], p0[1])
        p1 = (p1[0], p1[1])

        angle = np.arctan2(p1[1]-p0[1], p1[0]-p0[0])
        angles.append(angle)

        # K-means can't do periodic things like angles,
        # so translate angles to points on unit circle.
        # BUT, we want to cluster points near 0 and points
        # near pi together; so take double the angle
        unit_circle_positions.append((np.cos(2.*angle),
                                      np.sin(2.*angle)))
    km = KMeans(n_clusters=2)
    km.fit(unit_circle_positions)

    len0 = len(km.labels_) - sum(km.labels_)
    len1 = sum(km.labels_)
    if len1 > len0:
        group = 1
        small = 0
    else:
        group = 0
        small = 1

    ang = np.arctan2(km.cluster_centers_[group][1], km.cluster_centers_[group][0]) / 2. #divide by 2 bc we multiplied by 2 earlier
    return ang, km

        
