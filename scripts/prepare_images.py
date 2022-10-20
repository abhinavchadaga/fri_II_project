from pathlib import Path
import pickle
from argparse import ArgumentParser
import json

from detectron2.structures import BoxMode

import numpy as np
from skimage.draw import ellipse, disk, rectangle, polygon
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

    for building in Path(path_to_data).iterdir():
        cprint(f"\nprocessing {building} images...\n", "white", attrs=['bold'])
        # iterate through each building folder
        path_to_annotations = list(building.glob("*annotations.json"))[0]
        with path_to_annotations.open("r") as a:
            annotations = json.load(a)
            
        for k, a in annotations.items():
            # iterate through each annotated image
            fname = Path(a['filename'])
            fpath = building / fname

            with Image.open(fpath.as_posix()) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size

            # data required for one image
            data = {
                "file_name": fpath.as_posix(),
                "width": width,
                "height": height,
                "image_id": fname,
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
                elif shape_type == "circle":
                    rr, cc = disk(center=(s_attr['cy'], s_attr['cx']), radius=s_attr['r'])
                elif shape_type == "rect":
                    start = (s_attr['y'], s_attr['x'])
                    extent = (s_attr['height'], s_attr['width'])
                    rr, cc = rectangle(start=start, extent=extent)
                elif shape_type == "polygon":
                    r = s_attr["all_points_y"]
                    c = s_attr["all_points_y"]
                    rr, cc = polygon(r=r, c=c)
                elif shape_type == "polyline":
                    x = s_attr['all_points_x']
                    y = s_attr['all_points_y']
                    segmentation = [[]]
                    for i in range(len(x)):
                        segmentation[0].extend([x[i], y[i]])
                    bbox = [np.min(x), np.min(y), np.max(x), np.max(y)]
                else:
                    raise Exception(f"unsupported shape type: {shape_type}")

                if any([shape_type == "ellipse", shape_type == "circle", shape_type == "rect"]):
                    bbox = [np.min(cc), np.min(rr), np.max(cc), np.max(rr)]
                    segmentation = [[item for sublist in zip(cc, rr) for item in sublist]]
                
                annotation = {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": segmentation,
                    "category_id": category_id
                }
                data["annotations"].append(annotation)

            pkl_save_path = Path(save_path) / building.name
            pkl_save_path.mkdir(parents=True, exist_ok=True)
            pkl_save_path = pkl_save_path / (fname.stem + '.pkl')
            with pkl_save_path.open("wb") as f:
                pickle.dump(data, f)


        cprint(f"Finished {building} images!!\n", "green", attrs=['bold'])

cprint("\nDONE!!", "green", attrs=['bold'])