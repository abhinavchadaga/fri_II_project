import os
from glob import glob

if __name__ == "__main__":
    path = "/home/abhinavchadaga/cs/fri_II/final_project/datasets/labels/test"
    imgs = []
    for b in glob(os.path.join(path, "*")):
        for img in glob(os.path.join(b, "*")):
            for label in glob(os.path.join(img, "*")):
                l = f"{label} {label.split('/')[-1].split('.')[0]}\n"
                imgs.append(l)
    gt_file = os.path.join(path, "gtfile.txt")
    with open(gt_file, "w", encoding="utf-8") as f:
        f.writelines(imgs)

    print("done!!")
