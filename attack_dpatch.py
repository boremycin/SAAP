import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm

import yolov5

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import DPatch

from aap_utils.aap_utils import Yolo
from aap_utils.aap_utils import data_loader_test


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 1. Load YOLOv5
# -------------------------------------------------

model = yolov5.load("./weights/shipv5m.pt")
model = Yolo(model, alpha=0.1, beta=1)

detector = PyTorchYolo(
    model=model,
    device_type="gpu",
    input_shape=(3,640,640),
    clip_values=(0,255),
    attack_losses=("loss_total",)
)

# -------------------------------------------------
# Config
# -------------------------------------------------

test_img_num = 1000
src_path = "./dataset/clean_val/"
save_path = "./adv_examples/dpatch7/"
max_iter = 5000

# -------------------------------------------------
# Attack
# -------------------------------------------------

def generate_dpatch(source_path, save_path, iterations=200):

    attack = DPatch(
        estimator=detector,
        patch_shape=(3,120,120),
        learning_rate=5.0,
        max_iter=iterations,
        batch_size=1,
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

        # -------------------------
        # 2. Load image
        # -------------------------

        img = cv2.imread(str(img_path))
        img = cv2.resize(img,(640,640))

        img = img.transpose(2,0,1)        # HWC -> CHW
        x = np.expand_dims(img,axis=0).astype(np.float32)

        # -------------------------
        # 3. Generate patch
        # -------------------------

        patch = attack.generate(x=x,y=None)

        # -------------------------
        # 4. Apply patch
        # -------------------------

        x_adv = attack.apply_patch(x,patch_external=patch,random_location=True)

        # -------------------------
        # 5. Save image
        # -------------------------

        adv_img = x_adv[0].transpose(1,2,0)
        adv_img = np.nan_to_num(adv_img, nan=0.0, posinf=255.0, neginf=0.0)
        adv_img = np.clip(adv_img, 0, 255).astype(np.uint8)

        cv2.imwrite(str(save_name),adv_img)


# -------------------------------------------------

generate_dpatch(src_path,save_path,max_iter)