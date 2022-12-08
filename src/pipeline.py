import cv2
import numpy as np
import torch
from argparse import ArgumentParser

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Instances
from generate_pairs import get_nearest_k_btns

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from btn_label_assoc import BTNLabelAssoc
from prepare_data import resize_with_padding
from PIL import Image


if __name__ == "__main__":
    parser = ArgumentParser(description="run pipeline")
    parser.add_argument("--path_to_img", "-p", type=str, required=True)
    parser.add_argument("--floor", "-f", type=str, required=True)
    args = parser.parse_args()

    # CONFIGURE FIRST INSTANCE SEGMENTATION MODEL
    cascade_mask_rcnn_config = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cascade_mask_rcnn_config))
    cfg.DATASETS.TRAIN = ("elevators",)
    cfg.DATASETS.TEST = ()
    cfg.MODEL.WEIGHTS = (
        "/home/abhinavchadaga/cs/fri_II/final_project/trained_weights/model_final.pth"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # CONFIGURE MISSED DETECTIONS MODEL
    m_cfg = get_cfg()
    m_cfg.merge_from_file(model_zoo.get_config_file(cascade_mask_rcnn_config))
    m_cfg.DATASETS.TRAIN = ("elevators",)
    m_cfg.DATASETS.TEST = ()
    m_cfg.MODEL.WEIGHTS = (
        "/home/abhinavchadaga/cs/fri_II/final_project/trained_weights/missed_detections.pth"
    )
    m_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    m_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # CONFIGURE PARSEQ MODEL
    ckpt = (
        "/home/abhinavchadaga/cs/fri_II/final_project/trained_weights/parseq_trained_weights.ckpt"
    )

    # Load model and image transforms
    parseq = load_from_checkpoint(checkpoint_path=ckpt)
    img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

    # load button and label association model
    ckpt = "/home/abhinavchadaga/cs/fri_II/final_project/trained_weights/mlp_assoc_weights.ckpt"
    label_btns_model = BTNLabelAssoc.load_from_checkpoint(ckpt).model

    # LOAD IMAGE AND DO FIRST PASS
    im_path = args.path_to_img
    im = cv2.imread(im_path)
    im_height, im_width, _ = im.shape
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    # visualize output and write image
    v = Visualizer(im[:, :, ::-1], scale=1, instance_mode=ColorMode.SEGMENTATION)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("first_pass.jpg", out.get_image()[:, :, ::-1])

    instances = outputs["instances"]

    missed_predictor = DefaultPredictor(m_cfg)
    for i in range(len(instances)):
        pass

    # 0 is button 1 is label
    labels = []
    tgt_floor = args.floor
    label_indices = {}
    buttons = np.zeros_like(im)
    button_centers = []
    label_center = None

    lv = Visualizer(im[:, :, ::-1], scale=1, instance_mode=ColorMode.SEGMENTATION)
    for i in range(len(instances)):
        class_id = instances.pred_classes[i]
        if class_id == 1:
            bbox = (
                instances.pred_boxes[i]
                .tensor.detach()
                .to("cpu")
                .numpy()
                .squeeze(0)
                .astype(np.int32)
            )
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            label_arr = im[y1:y2, x1:x2, :].copy()
            label = Image.fromarray(label_arr)
            label = img_transform(label).unsqueeze(0)
            logits = parseq(label)
            pred = logits.softmax(-1)
            prediction, _ = parseq.tokenizer.decode(pred)
            label_indices[prediction[0]] = i
            labeled_img = lv.draw_text(prediction[0], (bbox[0], bbox[1]), color="g")
        else:
            bbox = (
                instances.pred_boxes[i]
                .tensor.detach()
                .to("cpu")
                .numpy()
                .squeeze(0)
                .astype(np.int32)
            )
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            button_centers.append(center)

    labeled_img = labeled_img.get_image()
    cv2.imwrite("labeled_val_img.jpg", labeled_img)

    index = label_indices.get(str(tgt_floor))
    if index is None:
        raise Exception("Label not detected!!")
    label_box = instances.pred_boxes[index].tensor.detach().to("cpu").squeeze(0).tolist()
    label_center = [(label_box[0] + label_box[2]) / 2, (label_box[1] + label_box[3]) / 2]
    nearest_btns = get_nearest_k_btns(label_center, button_centers)
    print(label_center)
    print(nearest_btns)
    input_vector = []
    for b in nearest_btns:
        input_vector.extend(b)
    input_vector.extend(label_center)
    input_vector = np.array(input_vector, dtype=np.float32)
    input_vector = torch.tensor(input_vector).unsqueeze(0)

    label_btns_model.eval()
    output = torch.exp(label_btns_model(input_vector))
    correct_btn = torch.argmax(output).item()
    print(torch.sum(output))
    print(output)
    print(nearest_btns[correct_btn])
    final_output = cv2.circle(
        im,
        (int(nearest_btns[correct_btn][0]), int(nearest_btns[correct_btn][1])),
        radius=5,
        color=(255, 0, 0),
        thickness=5,
    )
    cv2.imwrite("final_output.jpg", final_output)

    print("done!")
# cv2.imwrite(
#     f"/home/abhinavchadaga/cs/fri_II/final_project/labels/{prediction[0]}.jpg", label_arr
# )
#     if class_id == 0:
#         m = instances.pred_masks[i].detach().to("cpu").numpy()
#         m = np.expand_dims(m, axis=2)
#         m = np.concatenate([m, m, m], axis=2) * 255
#         buttons = buttons + m

# label_mask = instances.pred_masks[label_indices["1"]].detach().to("cpu").numpy()
# label_mask = label_mask * 255  # make red
# label_mask = np.expand_dims(label_mask, axis=2)
# bg = np.zeros_like(label_mask)
# label_mask = np.concatenate([label_mask, bg, bg], axis=2)
# print(label_mask.shape)

# association_input = (buttons + label_mask).astype(np.uint8)
# association_input = Image.fromarray(association_input)
# t = transforms.Compose([Resize((224, 224)), ToTensor()])
# association_input = t(association_input)
# association_input = torch.unsqueeze(association_input, dim=0)
# association_input = resize_with_padding(association_input, (224, 224))
# association_input = torch.tensor(association_input, dtype=torch.float32).unsqueeze(0)
# association_input = torch.transpose(association_input, 3, 1)

# print(association_input.shape)
# bbox = label_btns_model(association_input).squeeze(0)
# bbox = bbox.tolist()
# bbox = [
#     int(bbox[0] * im_width),
#     int(bbox[1] * im_height),
#     int(bbox[2] * im_width),
#     int(bbox[3] * im_height),
# ]

# min_dist = float("inf")
# min_index = -1
# for i in range(len(instances)):
#     class_id = instances.pred_classes[i]
#     if class_id != 0:
#         continue

#     box = instances.pred_boxes[i].tensor.squeeze(0).tolist()

#     distance = sum([(x[1] - x[0] ** 2) for x in zip(bbox, box)])
#     if distance < min_dist:
#         min_dist = distance
#         min_index = i

# btn_box = instances.pred_boxes[min_index].tensor.squeeze(0).type(torch.int32).tolist()
# print(bbox)
# correct_button = cv2.rectangle(
#     im, (btn_box[0], btn_box[1]), (btn_box[2], btn_box[3]), color=(255, 0, 0), thickness=5
# )
# print("done")
