import numpy as np
import torch
import cv2
import os
from pathlib import Path
from tqdm import tqdm

import yolov5

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import RobustDPatch

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
save_path = "./adv_examples/robustdpatch7/"
max_iter = 5000

# -------------------------------------------------
# Attack
# -------------------------------------------------

def generate_robustdpatch(source_path, save_path, iterations=200):

    attack = RobustDPatch(
        detector,
        patch_shape=(3,120,120),
        patch_location=(150,150),
        crop_range=(0,0),
        brightness_range=(0.7,1.4),
        rotation_weights=(0.7,0.1,0.1,0.1),
        sample_size=1,
        learning_rate=5.0,
        max_iter=iterations,
        batch_size=1,
        targeted=False,
        verbose=True
    )

    src_path = Path(source_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    img_list, txt_list, loc_list = data_loader_test(source_path, True, test_img_num)

    for i in tqdm(range(test_img_num)):

        stem = Path(txt_list[i]).stem
        img_path = src_path / "images" / f"{stem}.jpg"
        save_name = save_path / f"patched_{stem}.jpg"

        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (640, 640))

        img = img.transpose((2,0,1))
        x = np.expand_dims(img, axis=0).astype(np.float32)
        x_adv = attack.generate(x=x, y=None)

        adv_img = attack.apply_patch(x, patch_external=x_adv, random_location=True) 
        adv_img = np.nan_to_num(adv_img, nan=0.0, posinf=255.0, neginf=0.0)
        adv_img = np.clip(adv_img, 0, 255)
        adv_img = adv_img[0].transpose(1,2,0).astype(np.uint8)
        cv2.imwrite(str(save_name), adv_img)

# -------------------------------------------------

generate_robustdpatch(src_path, save_path, max_iter)