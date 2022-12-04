from argparse import ArgumentParser
import os
from glob import glob
import json
from typing import List

import numpy as np
import cv2
from termcolor import cprint

from data_utils import generate_gt_coords, generate_bbox

from PIL import Image, ImageOps


def resize_with_padding(img, expected_size):
    img = Image.fromarray(img)
    img.thumbnail((expected_size[0], expected_size[1]))
    # img = ImageOps.contain(img, expected_size)
    img = ImageOps.pad(img, expected_size)
    return np.asarray(img)


def process_images(path_to_data: str, save_dir: str) -> None:
    """generate labels and ground truths from each image in the dataset

    Args:
        path_to_data (str): path to elevator panel images and annotations
        save_dir (str): path to save cropped labels and ground truths
    """
    cprint(f"path to data: {path_to_data}", "white", attrs=["bold"])
    cprint(f"saving parsed data to: {save_dir}", "white", attrs=["bold"])
    print("\n")

    # get list of all buildlings
    pattern = os.path.join(path_to_data, "*")
    buildings: List[str] = glob(pattern)

    cprint("buildings:", color="cyan", attrs=["bold"])
    for b in buildings:
        print(b)
    print("\n")

    train_gt = []
    val_gt = []
    orig_save_dir = save_dir
    for b in buildings:
        if "mixed" in b or "inspire" in b and save_dir != os.path.join(orig_save_dir, "train"):
            save_dir = os.path.join(orig_save_dir, "train")
            gt_file = train_gt
        else:
            if save_dir != os.path.join(orig_save_dir, "val"):
                save_dir = os.path.join(orig_save_dir, "val")
                gt_file = val_gt

        os.makedirs(save_dir, exist_ok=True)
        # open annotations file
        path_to_annos = glob(os.path.join(b, "*annotations.json"))[0]
        annos: dict = json.load(open(path_to_annos))

        # process each image
        for img_dict in annos.values():
            img_fname = os.path.join(b, img_dict["filename"])
            img = cv2.imread(img_fname)

            for r in img_dict["regions"]:
                # skip non-labels
                if r["region_attributes"]["category_id"] != "label":
                    continue

                # skip labels with no gt value
                gt = r["region_attributes"].get("pair")
                if gt is None or gt == "":
                    continue

                if " " in gt:
                    raise Exception(img_fname)

                # make crop using bbox
                px, py = generate_gt_coords(r)
                bbox = generate_bbox(px=px, py=py)
                x1, y1, x2, y2 = bbox
                label = img[y1:y2, x1:x2, :].copy()
                label = resize_with_padding(label, (224, 224))

                # write image, add img save path and gt label to gt file
                filename = f"{img_fname.split('/')[-1].split('.')[0]}_{gt}.jpg"
                label_img_save_path = os.path.join(save_dir, filename)
                gt_file.append(f"{filename} {gt}\n")
                cv2.imwrite(label_img_save_path, label)

        b_name = b.split("/")[-1]
        cprint(f"finished {b_name}", "green")

    # write ground truth files
    train_save_path = os.path.join(orig_save_dir, "train")
    train_save_path = os.path.join(train_save_path, "gt.txt")
    with open(train_save_path, "w", encoding="utf-8") as f:
        f.writelines(train_gt)

    val_save_path = os.path.join(orig_save_dir, "val")
    val_save_path = os.path.join(val_save_path, "gt.txt")
    with open(val_save_path, "w", encoding="utf-8") as f:
        f.writelines(val_gt)

    cprint("success!!", "green", attrs=["bold"])


def main():
    parser = ArgumentParser(description="crop labels and save ground truths")
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
        default="/home/abhinavchadaga/cs/fri_II/final_project/data/labels/",
    )
    args = parser.parse_args()

    path_to_data = args.path_to_data
    save_path = args.save_path
    process_images(path_to_data=path_to_data, save_dir=save_path)


if __name__ == "__main__":
    main()
