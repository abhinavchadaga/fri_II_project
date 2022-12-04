from argparse import ArgumentParser
import os
from glob import glob
import pickle
from typing import List
import json
import math

import numpy as np
import cv2
from skimage.draw import ellipse_perimeter, circle_perimeter
from termcolor import cprint

from detectron2.structures import BoxMode

def generate_gt_coords(region: dict) -> List[int]:
    """ generate list of x coordinates (px) and y coordinates (py) for ground truth mask

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
        px = s_attr['all_points_x']
        py = s_attr['all_points_y']
    elif shape_type == "ellipse":
        rr, cc = ellipse_perimeter(r=int(s_attr['cy']),
                            c=int(s_attr['cx']),
                            r_radius=int(s_attr['ry']),
                            c_radius=int(s_attr['rx']),
                            orientation=math.radians(s_attr['theta']))
    elif shape_type == "circle":
        rr, cc = circle_perimeter(r=int(s_attr['cy']),
                                c=int(s_attr['cx']),
                                radius=int(s_attr['r']))
    elif shape_type == "rect":
        x, y = s_attr['x'], s_attr['y']
        width, height = s_attr['width'], s_attr['height']
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

def process_imgs(path_to_data: str, save_path: str) -> None:
    """ process all buildings' images in "path to data"
        creates pkl file for each image with dict of data conforming to Detectron2 specs

    Args:
        path_to_data (str): folder of buildings' images

            example:
                "path_to_data":
                    "building 1":
                        img_1.jpg
                        img_2.jpg
                        ...
                    "building 2":
                        img_1.jpg
                        img_2.jpg
                        ...

        save_path (str): location to save pkl files

            example:
                "save_path":
                    "building 1":
                        img_1.pkl
                        img_2.pkl
                        ...
                    "building 2":
                        img_1.pkl
                        img_2.pkl
                        ...
    """
    cprint(f"path to data: {path_to_data}", "white", attrs=["bold"])
    cprint(f"saving parsed data to: {save_path}", "white", attrs=["bold"])
    print("\n")

    # get list of all buildlings
    pattern = os.path.join(path_to_data, "*")
    buildings: List[str] = glob(pattern)

    cprint("buildings:", color="cyan", attrs=["bold"])
    for b in buildings:
        print(b)
    print("\n")

    # map classes to indices
    classes = {"button": 0, "label": 1, "not button": 2}
    image_id = 0

    for building in buildings:
        # open annotations file
        path_to_annos = glob(os.path.join(building, "*annotations.json"))[0]
        annos: dict = json.load(open(path_to_annos))

        # save directory for this building
        save_dir = os.path.join(save_path, building.split("/")[-1])
        os.makedirs(save_dir, exist_ok=True)

        # process each image
        for img in annos.values():
            record = {}

            filename = os.path.join(building, img["filename"])
            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width

            objs = []  # each feature in this image
            for r in img["regions"]:
                category_id = classes[r["region_attributes"]["category_id"]]
                px, py = generate_gt_coords(region=r)
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": category_id
                }

                objs.append(obj)

            # save image annotations into a pickle file
            record["annotations"] = objs
            record_save_path = filename.split(".")[0].split("/")[-1]
            record_save_path = os.path.join(save_dir, f"{record_save_path}.pkl")
            pickle.dump(record, open(record_save_path, "wb"))
            image_id += 1

        elevator_name = save_dir.split("/")[-1]
        cprint(f"finished {elevator_name}", "green")


    cprint("success!!", "green", attrs=["bold"])


def main():
    parser = ArgumentParser(description="convert images into pkl file for Detectron2")
    parser.add_argument("--path_to_data",
                        "-dp",
                        type=str,
                        default="/home/abhinavchadaga/cs/fri_II/final_project/data/"\
                                "elevator_panels")
    parser.add_argument("--save_path",
                        "-sp",
                        type=str,
                        default="/home/abhinavchadaga/cs/fri_II/final_project/datasets/"\
                                "elevator_panels")
    args = parser.parse_args()

    path_to_data = args.path_to_data
    save_path = args.save_path
    process_imgs(path_to_data=path_to_data, save_path=save_path)


if __name__ == "__main__":
    main()
