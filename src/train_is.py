# standard imports
from argparse import ArgumentParser
import os
from glob import glob
import pickle
from typing import List

# setup logger
from detectron2.utils.logger import setup_logger

# detectron2 utils
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog


def configure_dataset(path_to_dataset: str) -> List[dict]:
    def register_dataset():
        """register elevator panels dataset by
            loading all pkl files into memory

        Args:
            path_to_dataset (str): path to processed pkl files

                example:
                    "path_to_data":
                        "building 1":
                            img_1.jpg
                            img_2.jpg
                            ...
                        "building 2":
                            img_1.jpg
                            img_2.jpg
                            ...

        Returns:
            List[dict]: each image is represented by a dict in this list. dicts are in detectron2
                        format for datasets
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

    # register dataset
    DatasetCatalog.register("elevators", register_dataset)
    MetadataCatalog.get("elevators").thing_classes = ["button", "label", "not button"]


def configure_model(cfg: CfgNode, arch: str) -> None:
    CONFIG = (
        "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
        if arch == "cascade-rcnn"
        else "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.merge_from_file(model_zoo.get_config_file(CONFIG))
    cfg.DATASETS.TRAIN = ("elevators",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG)
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 20_000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3


def train_model(path_to_dataset: str, arch: str):
    # register dataset
    configure_dataset(path_to_dataset=path_to_dataset)

    # create model configuration
    cfg = get_cfg()
    configure_model(cfg, arch=arch)

    # train model using DefaultTrainer
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main():
    setup_logger()
    parser = ArgumentParser(description="train instance segmentation model")
    parser.add_argument(
        "--path_to_dataset",
        "-p",
        type=str,
        default="/home/abhinavchadaga/cs/fri_II/final_project/datasets/" "panels",
    )
    parser.add_argument("--arch", "-a", choices=["cascade-rcnn", "mrcnn"], default="cascade-rcnn")
    args = parser.parse_args()
    train_model(path_to_dataset=args.path_to_dataset, arch=args.arch)


if __name__ == "__main__":
    main()
