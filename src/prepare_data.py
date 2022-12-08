from argparse import ArgumentParser
import os
from glob import glob
import pickle
from typing import List
import json
from multiprocessing import Pool

import cv2
import numpy as np
from termcolor import cprint
from PIL import Image, ImageOps

from detectron2.structures import BoxMode
from data_utils import generate_gt_coords, generate_bbox, generate_opaque_mask


def get_pair_gt(r: dict):
    gt: str = r["region_attributes"].get("pair")
    if gt is None or gt == "":
        return None

    if " " in gt:
        gt.replace(" ", "_")

    return gt


def resize_with_padding(array, expected_size):
    img = Image.fromarray(array)
    img.thumbnail((expected_size[0], expected_size[1]))
    img = ImageOps.pad(img, expected_size)
    return np.asarray(img)


def generate_d2_obj(r: dict):
    classes = {"button": 0, "label": 1, "not button": 2}
    px, py = generate_gt_coords(region=r)
    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
    poly = [p for x in poly for p in x]
    bbox = generate_bbox(px=px, py=py)

    return {
        "bbox": bbox,
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [poly],
        "category_id": classes[r["region_attributes"]["category_id"]],
        "pair": r["region_attributes"].get("pair"),
    }


def generate_label_crop(orig_img: cv2.Mat, crop_box: List[int], r: dict, save_dir: str):
    # crop and resize label img
    x1, y1, x2, y2 = crop_box
    label = orig_img[y1:y2, x1:x2, :].copy()
    label = resize_with_padding(label, (224, 224))

    gt = get_pair_gt(r)

    # write label to disk
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{gt}.jpg")
    cv2.imwrite(save_path, label)


def generate_buttons_seg_mask(img_dict: dict, orig_img: cv2.Mat):
    # add all btn segmentation masks to a single image
    buttons = np.zeros_like(orig_img)
    height, width, _ = orig_img.shape
    for r in img_dict["regions"]:
        category_id = r["region_attributes"]["category_id"]
        if category_id == "label":
            continue

        rr, cc = generate_opaque_mask(r, height, width)
        buttons[rr, cc, :] = (255, 255, 255)

    return buttons


def generate_lba_img(
    label: dict, buttons: cv2.Mat, orig_img: cv2.Mat, gt_bbox: List[int], save_dir: str
):
    # generate segmentation mask for label in RED
    # add it to buttons
    assoc_img = np.zeros_like(orig_img)
    height, width, _ = orig_img.shape
    rr, cc = generate_opaque_mask(label, height, width)
    assoc_img[rr, cc, :] = (0, 0, 255)
    assoc_img += buttons
    # assoc_img = resize_with_padding(assoc_img, (224, 224))

    gt_bbox = [
        float(gt_bbox[0]) / width,
        float(gt_bbox[1]) / height,
        float(gt_bbox[2]) / width,
        float(gt_bbox[3]) / height,
    ]

    gt = get_pair_gt(label)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{gt}.pkl")
    data = {"img": assoc_img, "gt": gt_bbox}
    with open(save_path, "wb") as f:
        pickle.dump(data, f)


def process_imgs(buildings: str, panels: str, labels: str, label_btns_assoc: str) -> None:
    classes = {"button": 0, "label": 1, "not button": 2}
    image_id = 0
    # map classes to indices
    buildings = sorted(buildings)
    for b in buildings:
        # open annotations file
        path_to_annos = glob(os.path.join(b, "*annotations.json"))[0]
        annos: dict = json.load(open(path_to_annos))

        # building name
        bname = b.split("/")[-1]

        # process each image
        for i, img_dict in enumerate(annos.values()):
            record = {}

            filename = img_dict["filename"]
            img_path = os.path.join(b, filename)
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            height, width, _ = img.shape
            record["file_name"] = img_path
            record["image_id"] = image_id
            record["height"] = height
            record["width"] = width

            objs = []  # each feature in this image

            for r in img_dict["regions"]:
                # GENERATE MASK
                obj = generate_d2_obj(r)
                objs.append(obj)

                # GENERATE LABEL
                if r["region_attributes"]["category_id"] == "label":
                    generate_label_crop(
                        orig_img=img,
                        crop_box=obj["bbox"],
                        r=r,
                        save_dir=os.path.join(labels, bname, str(i)),
                    )

            # save image annotations into a pickle file
            record["annotations"] = objs
            record_save_dir = os.path.join(panels, bname)
            os.makedirs(record_save_dir, exist_ok=True)
            record_save_path = os.path.join(record_save_dir, f"{i}.pkl")
            with open(record_save_path, "wb") as f:
                pickle.dump(record, f)

            # GENERATE LABEL BUTTON ASSSOCIATION NETWORK INPUT IMGS
            # for each label, show label and all buttons

            # buttons = generate_buttons_seg_mask(img_dict=img_dict, orig_img=img)
            # for r in img_dict["regions"]:
            #     category_id = r["region_attributes"]["category_id"]
            #     if category_id == "label":
            #         pair = r["region_attributes"].get("pair")
            #         if pair is None:
            #             continue
            #         gt_bbox = None
            #         # look for associated button
            #         for o in objs:
            #             if o["category_id"] == 0 and o["pair"] == pair:
            #                 gt_bbox = o["bbox"]
            #                 break

            #         # no button for this label
            #         if gt_bbox == None:
            #             continue

            #         generate_lba_img(
            #             label=r,
            #             buttons=buttons,
            #             orig_img=img,
            #             gt_bbox=gt_bbox,
            #             save_dir=os.path.join(label_btns_assoc, bname, str(i)),
            #         )
            image_id += 1
            # print(f"done {filename}")

        cprint(f"finished {bname}", "green")

    cprint("success!!", "green", attrs=["bold"])


def main():
    parser = ArgumentParser(description="convert images into pkl file for Detectron2")
    parser.add_argument(
        "--path_to_data",
        "-dp",
        type=str,
        default="/home/abhinavchadaga/cs/fri_II/final_project/data/panels/test",
    )
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        default="/home/abhinavchadaga/cs/fri_II/final_project/datasets/panels/test",
    )
    args = parser.parse_args()

    path_to_data = args.path_to_data
    panels_save_path = args.save_path
    labels_save_path = "/home/abhinavchadaga/cs/fri_II/final_project/datasets/labels/test"
    lb_assoc_save_path = "/home/abhinavchadaga/cs/fri_II/final_project/datasets/association_set"

    cprint(f"path to data: {path_to_data}", "white", attrs=["bold"])
    cprint(f"saving panels data to: {panels_save_path}", "white", attrs=["bold"])
    cprint(f"saving labels data to: {labels_save_path}", "white", attrs=["bold"])
    cprint(
        f"saving label button association inputs data to: {lb_assoc_save_path}",
        "white",
        attrs=["bold"],
    )
    print("\n")

    # get list of all buildlings
    pattern = os.path.join(path_to_data, "*")
    buildings: List[str] = glob(pattern)

    cprint("buildings:", color="cyan", attrs=["bold"])
    for b in buildings:
        print(b)
    print("\n")

    process_imgs(
        buildings=buildings,
        panels=panels_save_path,
        labels=labels_save_path,
        label_btns_assoc=lb_assoc_save_path,
    )


if __name__ == "__main__":
    main()
