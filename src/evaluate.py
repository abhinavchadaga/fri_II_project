from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from train_is import configure_dataset

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

configure_dataset("/home/abhinavchadaga/cs/fri_II/final_project/datasets/panels/test/")

CONFIG = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
WEIGHTS = "/home/abhinavchadaga/cs/fri_II/final_project/trained_weights/model_final.pth"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG))
cfg.DATASETS.TRAIN = ()
cfg.DATASETS.TEST = "elevators"
cfg.DATALOADER.NUM_WORKERS = 12
cfg.MODEL.WEIGHTS = WEIGHTS
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("elevators", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "elevators")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
