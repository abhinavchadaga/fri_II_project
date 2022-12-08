import math
import os
from argparse import ArgumentParser
from typing import List
from glob import glob
import pickle

import numpy as np
from termcolor import cprint

check = []


def get_nearest_k_btns(label, btn_centers, btn_pairs=None, k=4):
    if btn_pairs is None:
        top_4_btn_centers = sorted(btn_centers, key=lambda b: math.dist(label, b))[:k]
        return top_4_btn_centers
    else:
        top_4_btn_centers, top_4_pairs = zip(
            *sorted(zip(btn_centers, btn_pairs), key=lambda b: math.dist(label, b[0]))
        )
        return top_4_btn_centers[:k], top_4_pairs[:k]


def print_error_message(building, img_dict, pair):
    print(f"{building.split('/')[-1]}, {img_dict['file_name'].split('/')[-1]}, {pair}")


def generate_pairs(path_to_data: str, save_dir: str) -> None:
    cprint(f"path to data: {path_to_data}", "white", attrs=["bold"])
    cprint(f"saving parsed data to: {save_dir}", "white", attrs=["bold"])
    print("\n")

    # get list of all buildlings
    pattern = os.path.join(path_to_data, "*")
    buildings: List[str] = glob(pattern)

    cprint("buildings:", color="cyan", attrs=["bold"])
    for building in buildings:
        print(building)
    print("\n")

    for building in buildings:
        # get list of pkl files in this building directory
        pkl_files = glob(os.path.join(building, "*.pkl"))
        for p in pkl_files:
            # open pkl file
            with open(p, "rb") as f:
                img_dict = pickle.load(f)

                # get list of buttons, labels, and their corresponding classes
                btn_centers = []
                label_centers = []
                label_pairs = []
                btn_pairs = []
                for a in img_dict["annotations"]:
                    bbox = a["bbox"]
                    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    if a["category_id"] in (0, 2):
                        btn_centers.append(center)
                        btn_pairs.append(a.get("pair"))
                    elif a["category_id"] == 1:
                        label_centers.append(center)
                        label_pairs.append(a.get("pair"))

                for label, pair in zip(label_centers, label_pairs):
                    if pair == "" or pair is None:
                        continue

                    top_4_btn_centers, top_4_pairs = get_nearest_k_btns(
                        label, btn_centers, btn_pairs, k=4
                    )

                    # generate input vector
                    input_vec = []
                    for btn in top_4_btn_centers:
                        input_vec.extend(btn)
                    if len(input_vec) != 8:
                        check.append(f"{img_dict['file_name'].split('/')[-1]}, {pair}")
                        check.append(f"\t{top_4_pairs}")
                        continue
                    input_vec.extend(label)
                    input_vec = np.array(input_vec)

                    gt = None

                    for i, p in enumerate(top_4_pairs):
                        if p == pair:
                            gt = i

                    if gt is None or gt >= 4:
                        check.append(f"{img_dict['file_name'].split('/')[-1]}, {pair}")
                        check.append(f"\t{top_4_pairs}")
                        continue

                    gt = np.array(gt)
                    data = {"input": input_vec, "gt": gt, "filename": p, "pair": pair}
                    save_path = os.path.join(save_dir, building.split("/")[-1])
                    save_path = os.path.join(
                        save_path, img_dict["file_name"].split("/")[-1].split(".")[0]
                    )
                    os.makedirs(save_path, exist_ok=True)
                    # add file name
                    save_path = os.path.join(save_path, f"{pair}.pkl")
                    with open(save_path, "wb") as f:
                        pickle.dump(data, f)

        cprint(f"finished {building.split('/')[-1]}", "green")


def main():
    parser = ArgumentParser(description="crop labels and save ground truths")
    parser.add_argument(
        "--path_to_data",
        "-dp",
        type=str,
        default="/home/abhinavchadaga/cs/fri_II/final_project/datasets/panels",
    )
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        default="/home/abhinavchadaga/cs/fri_II/final_project/datasets/pairs",
    )
    args = parser.parse_args()

    path_to_data = args.path_to_data
    save_path = args.save_path
    generate_pairs(path_to_data=path_to_data, save_dir=save_path)
    cprint("success!!", "green", attrs=["bold"])

    cprint("check these: ", "red")
    for i in range(0, len(check), 2):
        # mixed errors can be skipped
        c = check[i]
        if "mixed" not in c and not ("signature" in c and "call" in c):
            print(c)
            print(check[i + 1])


if __name__ == "__main__":
    main()
