from argparse import ArgumentParser
import os
from glob import glob
import pickle
from typing import List

# setup logger
from detectron2.utils.logger import setup_logger
setup_logger()

# detectron2 utils
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

path_to_dataset = "datasets/elevator_panels"

def register_dataset(path_to_dataset: str) -> List[dict]:
    """ register elevator panels dataset by 
        loading all pkl files into memory
    """
    elevators = glob(f"{path_to_dataset}/*")
    dataset: List[dict] = []
    for e in elevators:
        imgs = glob(f"{e}/*")
        for im in imgs:
            with open(im, "rb") as f:
                d = pickle.load(f)
                dataset.append(d)
    return dataset


if __name__ == "__main__":
    parser = ArgumentParser(description="train instance segmentation model")
    parser.add_argument("--path_to_dataset", "-p", type=str, default="/home/abhinavchadaga/cs/fri_II/final_project/datasets/elevator_panels")
    parser.add_argument("--arch", "-a", choices=["cascade-rcnn", "mrcnn"], default="cascade-rcnn")
    args = parser.parse_args()

    path_to_dataset = args.path_to_dataset
    config_file = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml" if args.arch == "cascade-rcnn" else "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    # register dataset
    DatasetCatalog.register("elevators", lambda: register_dataset(path_to_dataset))
    MetadataCatalog.get("elevators").thing_classes = ["button", "label", "not button"]

    dataset: List[dict] = DatasetCatalog.get("elevators")

    # configure model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.DATASETS.TRAIN = ("elevators", )
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10_000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # train model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()