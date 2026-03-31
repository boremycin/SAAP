import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import yolov5

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import AdversarialPatchPyTorch

from aap_utils.aap_utils import Yolo
from aap_utils.aap_utils import data_loader_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load YOLOv5
# -------------------------------------------------

model = yolov5.load("./weights/shipv5m.pt")
model = Yolo(model, alpha=0.1, beta=1)

detector = PyTorchYolo(
    model=model,
    device_type="gpu",
    input_shape=(3, 640, 640),
    clip_values=(0, 255),
    attack_losses=("loss_total",)
)

# -------------------------------------------------
# Config
# -------------------------------------------------

test_img_num = 1000
src_path = "./dataset/clean_val/"
save_path = "./adv_examples/advpatch7/"
max_iter = 5000

# -------------------------------------------------
# Convert YOLO prediction to ART format
# -------------------------------------------------

def filter_boxes(predictions, conf_thresh=0.5):

    boxes_list = []
    scores_list = []
    labels_list = []

    for i in range(len(predictions[0]["boxes"])):

        score = predictions[0]["scores"][i]

        if score >= conf_thresh:
            boxes_list.append(predictions[0]["boxes"][i])
            scores_list.append(predictions[0]["scores"][i])
            labels_list.append(predictions[0]["labels"][i])

    if len(boxes_list) == 0:
        return [{
            "boxes": np.zeros((0,4)),
            "scores": np.zeros((0,)),
            "labels": np.zeros((0,))
        }]

    y = [{
        "boxes": np.array(boxes_list),
        "scores": np.array(scores_list),
        "labels": np.array(labels_list)
    }]

    return y


# -------------------------------------------------
# Attack
# -------------------------------------------------

def generate_advpatch(source_path, save_path, iterations=200):

    src_path = Path(source_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    img_list, txt_list, loc_list = data_loader_test(source_path, True, test_img_num)

    for i in tqdm(range(test_img_num)):
        attack = AdversarialPatchPyTorch(
            estimator=detector,
            rotation_max=0.0,
            scale_min=0.1,
            scale_max=0.3,
            learning_rate=2.0,
            max_iter=iterations,
            batch_size=1,
            patch_shape=(3, 200, 200),
            patch_type="square",
            targeted=False,
            verbose=False
        )

        stem = Path(txt_list[i]).stem
        img_path = src_path / "images" / f"{stem}.jpg"
        save_name = save_path / f"patched_{stem}.jpg"

        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (640, 640))

        img = img.transpose((2,0,1))
        x = np.expand_dims(img, axis=0).astype(np.float32)
        preds = detector.predict(x)
        y = filter_boxes(preds)
        patch, _ = attack.generate(x=x, y=y)

        x_adv = attack.apply_patch(
            x=x,
            scale=0.15,
            patch_external=patch
        )

        adv_img = x_adv[0].transpose(1,2,0)

        cv2.imwrite(str(save_name), adv_img)


# -------------------------------------------------
generate_advpatch(src_path, save_path, max_iter)