from pathlib import Path
import pickle
from argparse import ArgumentParser
import json

from detectron2.structures import BoxMode

import numpy as np
from skimage.draw import ellipse
from PIL import Image, ImageOps
from termcolor import cprint

""" Process all images for a given building
return a pickle file for each image with information
matching the dict required for registering a custom dataset
with detectron2
"""

if __name__ == "__main__":
    parser = ArgumentParser(description="process all input data")
    parser.add_argument("--data", "-d", type=str,
                        default="/home/abhinavchadaga/CS/fri_II/data/building_images")
    parser.add_argument("--save_path", "-s", type=str,
                        default="/home/abhinavchadaga/CS/fri_II/dataset/annotated_building_imgs")

    args = parser.parse_args()
    path_to_data = args.data
    save_path = args.save_path

    cprint(f"loading building data from: {path_to_data}", "white", attrs=['bold'])
    cprint(f"saving annotated images to: {save_path}\n", "white", attrs=['bold'])

    classes = {'label': 0, 'button': 1, 'not button': 2}
    cprint(f"using classes:", "white", attrs=['bold'])
    for k in classes.keys():
        print(f"\t{k}")

    image_id = 0
    for building in Path(path_to_data).iterdir():
        cprint(f"\nprocessing {building} images...\n", "white", attrs=['bold'])
        # iterate through each building folder
        path_to_annotations = list(Path(building).glob("*annotations.json"))[0].as_posix()
        annotations = json.load(open(path_to_annotations))
        for k, a in annotations.items():
            # iterate through each annotated image
            file_name = Path(a['filename'])
            full_path = (building / file_name).as_posix()

            with Image.open(full_path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size

            cprint(f"loaded {full_path}", "green")

            # data required for one image
            data = {
                "file_name": full_path,
                "width": width,
                "height": height,
                "image_id": image_id,
                "annotations": []
            }

            for instance in a["regions"]:
                # iterate through each annotation in this image
                r_attr = instance["region_attributes"]
                s_attr = instance["shape_attributes"]
                category_id = classes[r_attr["category_id"]]
                shape_type = s_attr["name"]

                # get list of coordinates for this annotation
                if shape_type == "ellipse":
                    rr, cc = ellipse(r=s_attr['cy'],
                                     c=s_attr['cx'],
                                     r_radius=s_attr['ry'],
                                     c_radius=s_attr['rx'],
                                     rotation=s_attr['theta'])
                    bbox = [np.min(cc), np.min(rr), np.max(cc), np.max(rr)]
                    segmentation = [[item for sublist in zip(cc, rr) for item in sublist]]
                elif shape_type == "polyline":
                    x = s_attr['all_points_x']
                    y = s_attr['all_points_y']
                    segmentation = [[]]
                    for i in range(len(x)):
                        segmentation[0].extend([x[i], y[i]])
                    bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]
                else:
                    bbox = None
                    segmentation = None

                annotation = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": segmentation,
                    "category_id": category_id
                }
                data["annotations"].append(annotation)

            pkl_save_path = Path(save_path) / building.name
            pkl_save_path.mkdir(parents=True, exist_ok=True)
            pkl_save_path = pkl_save_path / Path(f"{file_name.as_posix().split('.')[0]}.pkl")
            with pkl_save_path.open("wb") as f:
                pickle.dump(data, f)

            cprint(f"saved {full_path} to {pkl_save_path}!!", "green", attrs=['bold'])
            image_id += 1

cprint("\nDONE!!", "green", attrs=['bold'])