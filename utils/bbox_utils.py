import sympy
from sympy import Line 

def get_center_of_bbox(bbox): # gets center of bounding box and returns as tuple
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)
def measure_distance(p1, p2): # measures distance between two points 
    return ((p1[0] - p2[0])**2 + (p1[1]-p2[1])**2)**0.5
def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], sympy.geometry.point.Point2D):
            point = intersection[0].coordinates
    return point