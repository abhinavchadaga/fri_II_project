from typing import List
import glob
import os
import pickle

from termcolor import cprint
import torch

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultTrainer

path_to_dataset = "/home/abhinavchadaga/CS/fri_II/dataset/annotated_building_imgs"

dirs = glob.glob(os.path.join(path_to_dataset, "*"))
print(dirs)
num_buildings = len(dirs)
trainset_buildings = []
testset_buildings = []
trainset_size = int(0.7 * num_buildings)
testset_size = num_buildings - trainset_size

for i, b in enumerate(dirs):
    if i < trainset_size:
        trainset_buildings.append(b)
    else:
        testset_buildings.append(b)

cprint(f"trainset size: {trainset_size}", "green", attrs=["bold"])
cprint(f"testset size: {testset_size}\n", "green", attrs=["bold"])

print(f"trainset: ")
for b in trainset_buildings:
    print(b)
print()

print(f"testset: ")
for b in testset_buildings:
    print(b)


# def register_dataset(buildings: List[str]):
#     """ Define function to register 
#         our custom dataset using the specified buildings
#     """
#     dataset = list()
#     for building in buildings:
#         for img_path in os.listdir(building):
#             with open(os.path.join(building, img_path), "rb") as img:
#                 dataset.append(pickle.load(img))

#     return dataset


# thing_classes = ["label", "button", "not button"]

# DatasetCatalog.register("train", lambda: register_dataset(trainset_buildings))
# MetadataCatalog.get("train").thing_classes = thing_classes

# DatasetCatalog.register("test", lambda: register_dataset(testset_buildings))
# MetadataCatalog.get("test").thing_classes = thing_classes

# device = "cuda" if torch.cuda.is_available else "cpu"
# cprint(f"using {device}", "red", attrs=["bold"])

# cfg = get_cfg()
# cfg.merge_from_file(
#     model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.FREEZE_AT = 5
# cfg.DATASETS.TRAIN = ("train", )
# cfg.DATASETS.TEST = ("test",)
# cfg.DATALOADER.NUM_WORKERS = 8
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# )  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
# cfg.SOLVER.BASE_LR = 3e-5  # pick a good LR
# cfg.SOLVER.MAX_ITER = 8000
# cfg.SOLVER.STEPS = []  # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3



# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=True)
# trainer.train()

# cprint("trained and saved model!", "green", attrs=["bold"])