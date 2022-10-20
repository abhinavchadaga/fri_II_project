from argparse import ArgumentParser
import json
from pathlib import Path

from termcolor import cprint
""" Simple script to rename all files and folders to a standard format
    modifies the contents of the annotations to match

    Desired Format:

    data:
        building_images:
            building_x:
                building_x_1
                building_x_2
                ...
                building_x_annotations.json
            building_y:
                building_y_1
                building_y_2
                ...
                building_y_annotations.json                
"""

if __name__ == "__main__":
    parser = ArgumentParser(description="standardize file and folder naming")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default="/home/abhinavchadaga/CS/fri_II/data/building_images")
    args = parser.parse_args()
    path_to_data = Path(args.data)

    cprint("Renaming files in the following directories: ",
           "white",
           attrs=["bold"])

    for building_dir in path_to_data.iterdir():
        print(building_dir)
        img_paths = [
            p.resolve() for p in building_dir.glob("**/*")
            if p.suffix in {".jpeg", ".JGP", ".png", ".jpg"}
        ]
        annotation_path = list(building_dir.glob("*.json"))[0]
        with annotation_path.open("r") as a:
            data = json.load(a)

        annotations = {}
        for i, (k, v) in enumerate(data.items()):
            fname = Path(v['filename'])
            fpath = building_dir / fname

            new_fname = f"{building_dir.name}_{i}{fpath.suffix}"
            renamed_key = f"{new_fname}{k[len(fname.as_posix()):]}"

            annotations[renamed_key] = v
            annotations[renamed_key]['filename'] = new_fname

            fpath.rename(Path(fpath.parent, new_fname))

        with annotation_path.open("w") as a:
            json.dump(annotations, a)

    cprint("\nsucessfully renamed all files!!", "green", attrs=["bold"])