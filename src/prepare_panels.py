from argparse import ArgumentParser
import os
from glob import glob
import pickle
from typing import List
import json

import cv2
from termcolor import cprint

from detectron2.structures import BoxMode
from data_utils import generate_gt_coords, generate_bbox


def process_imgs(path_to_data: str, save_path: str) -> None:
    """process all buildings' images in "path to data"
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

    for b in buildings:
        # open annotations file
        path_to_annos = glob(os.path.join(b, "*annotations.json"))[0]
        annos: dict = json.load(open(path_to_annos))

        # save directory for this building
        b_save_dir = os.path.join(save_path, b.split("/")[-1])
        os.makedirs(b_save_dir, exist_ok=True)

        # process each image
        for img_dict in annos.values():
            record = {}

            filename = os.path.join(b, img_dict["filename"])
            height, width = cv2.imread(filename).shape[:2]
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width

            objs = []  # each feature in this image
            for r in img_dict["regions"]:
                category_id = classes[r["region_attributes"]["category_id"]]
                px, py = generate_gt_coords(region=r)
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                bbox = generate_bbox(px=px, py=py)

                obj = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": category_id,
                }

                objs.append(obj)

            # save image annotations into a pickle file
            record["annotations"] = objs
            record_save_path = filename.split(".")[0].split("/")[-1]
            record_save_path = os.path.join(b_save_dir, f"{record_save_path}.pkl")
            with open(record_save_path, "wb") as f:
                pickle.dump(record, f)

            image_id += 1

        b_name = b_save_dir.split("/")[-1]
        cprint(f"finished {b_name}", "green")

    cprint("success!!", "green", attrs=["bold"])


def main():
    parser = ArgumentParser(description="convert images into pkl file for Detectron2")
    parser.add_argument(
        "--path_to_data",
        "-dp",
        type=str,
        default="/home/abhinavchadaga/cs/fri_II/final_project/data/panels",
    )
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        default="/home/abhinavchadaga/cs/fri_II/final_project/datasets/panels",
    )
    args = parser.parse_args()

    path_to_data = args.path_to_data
    save_path = args.save_path
    process_imgs(path_to_data=path_to_data, save_path=save_path)


if __name__ == "__main__":
    main()
