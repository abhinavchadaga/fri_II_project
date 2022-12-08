import math
from typing import List

import numpy as np
from skimage.draw import ellipse_perimeter, circle_perimeter, disk, rectangle, ellipse, polygon


def generate_opaque_mask(region: dict, height: int, width: int):
    s_attr = region["shape_attributes"]
    shape_type = s_attr["name"]

    # px -> list of x values
    # py -> list of y values
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if shape_type == "circle":
        rr, cc = disk(center=(s_attr["cy"], s_attr["cx"]), radius=s_attr["r"])
    elif shape_type == "rect":
        start = (s_attr["y"], s_attr["x"])
        extent = (s_attr["height"], s_attr["width"])
        rr, cc = rectangle(start=start, extent=extent)
    elif shape_type == "ellipse":
        rr, cc = ellipse(
            r=s_attr["cy"],
            c=s_attr["cx"],
            r_radius=s_attr["ry"],
            c_radius=s_attr["rx"],
            rotation=np.deg2rad(s_attr["theta"]),
        )
    else:
        rr, cc = polygon(r=s_attr["all_points_y"], c=s_attr["all_points_x"])

    return rr, cc




def generate_gt_coords(region: dict) -> List[int]:
    """generate list of x coordinates (px) and y coordinates (py) for ground truth mask

    region format:

    {
        "shape_attributes": {
            "name": ...
            ...
            ...
        },

        "region attributes": {
            "category_id": ...
            "pair": ...
        }
    }

    """
    s_attr = region["shape_attributes"]
    shape_type = s_attr["name"]

    # px -> list of x values
    # py -> list of y values
    if shape_type in ("polyline", "polygon"):
        px = s_attr["all_points_x"]
        py = s_attr["all_points_y"]
    elif shape_type == "ellipse":
        rr, cc = ellipse_perimeter(
            r=int(s_attr["cy"]),
            c=int(s_attr["cx"]),
            r_radius=int(s_attr["ry"]),
            c_radius=int(s_attr["rx"]),
            orientation=math.radians(s_attr["theta"]),
        )
    elif shape_type == "circle":
        rr, cc = circle_perimeter(r=int(s_attr["cy"]), c=int(s_attr["cx"]), radius=int(s_attr["r"]))
    elif shape_type == "rect":
        x, y = s_attr["x"], s_attr["y"]
        width, height = s_attr["width"], s_attr["height"]
        topleft = (x, y)
        topright = (x + width, y)
        bottomleft = (x, y + height)
        bottomright = (x + width, y + height)
        px = [topleft[0], bottomleft[0], bottomright[0], topright[0]]
        py = [topleft[1], bottomleft[1], bottomright[1], topright[1]]
    else:
        raise Exception("unsupported shape type")

    # sort px, py for ellipse and circle
    if shape_type in ("circle", "ellipse"):
        angle = np.arctan2(rr - np.mean(rr), cc - np.mean(cc))
        sorted_by_angle = np.argsort(angle)
        py = rr[sorted_by_angle]
        px = cc[sorted_by_angle]

    return px, py


def generate_bbox(px: List[int], py: List[int]) -> List[int]:
    return [np.min(px), np.min(py), np.max(px), np.max(py)]
