from argparse import ArgumentParser
import os
import json

import sys
import cv2

from instance_segmentation.prepare_elevator_panels import generate_gt_coords

if __name__ == "__main__":
    parser = ArgumentParser(description="generate label dataset, individual labels + gt")
    parser.add_argument("--path_to_data", "-p", type=str, default="./data/elevator_panels")
    args = parser.parse_args()

    # get path to data
    path_to_data = args.path_to_data

    print(os.getcwd())

    img_path = "/home/abhinavchadaga/cs/fri_II/final_project/data/elevator_panels/ahg/ahg_8.JPG"
    img: cv2.Mat = cv2.imread(img_path)

    annos_path = "/home/abhinavchadaga/cs/fri_II/final_project/data/elevator_panels/ahg/"\
                 "ahg_annotations.json"
    annos: dict = json.load(open(annos_path))

    anno = None
    num_labels = 0
    for k, v in annos.items():
        if v["filename"] == os.path.basename(os.path.normpath(img_path)):
            anno = v
            # calculate num labels
            for r in v["regions"]:
                if r["region_attributes"]["category_id"] == "label":
                    num_labels += 1
            break

    print(anno["filename"])
    print(f"num labels: {num_labels}")
