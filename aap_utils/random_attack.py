from pathlib import Path
import numpy as np
from sympy import false
import torch
import numpy as np
from torchvision.transforms import transforms
from art.attacks.poisoning import perturbations
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
import os
import cv2
import sys
sys.path.append('./aap_utils')
from aap_utils.aap_utils import Yolo
from aap_utils.aap_utils import data_loader_test
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import yolov5
from tqdm import tqdm
#--------------------------------------------------------------------

# Get Yolov5 model and corresponding detector class
model = yolov5.load("./weights/shipv5m.pt")
model = Yolo(model,alpha = 0.1,beta = 1) #control the proportion of different loss
#model.to(device)  

#discard the class loss
detector = PyTorchYolo(
    model=model, device_type="gpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
)
#--------------------------------------------------------------------

NUMBER_CHANNELS = 3
INPUT_SHAPE = (NUMBER_CHANNELS, 256, 256)

transform = transforms.Compose([
        transforms.Resize(INPUT_SHAPE[1], interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(INPUT_SHAPE[1]),
        transforms.ToTensor()
    ])

def whole_random(source_path, save_path, eps = 2/255,test_img_num = 100,attenuation = False):
    src_path = Path(source_path)
    ad_path = Path(save_path) 
    ad_path.mkdir(parents=True, exist_ok=True)

    img_list,txt_list,loc_list = data_loader_test(source_path, True, test_img_num) #The loc_list stores the geometric center points of the ships after the images have been scaled down to 640.
    
    for i in tqdm(range(test_img_num), desc="Processing whole_random images", position=0, leave=True):
        txt_path = Path(txt_list[i])
        file_stem = txt_path.stem

        img_file_path = src_path / "images" / f"{file_stem}.jpg"
        img_name = source_path + 'images/' +(txt_list[i].split('/')[-1])[:-3] + 'jpg'
        img_save_name = 'patched_' + (txt_list[i].split('/')[-1])[:-3] + 'jpg'
        
        img_raw = cv2.imread(img_file_path.as_posix())
        img = cv2.imread(img_file_path.as_posix())
        if img_raw is None:
            print(f"Warning: Could not load image {img_name}")
            continue

        img = cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR) #Image scaling interpolation method utilized by yolov5
        img_reshape = img.transpose((2, 0, 1)).astype(np.float32) #channel,width,height

        perturb = np.random.normal(0, eps, img_reshape.shape).astype(np.float32)
        if attenuation:
            img_adv = img_reshape - perturb*255.0
            img_adv = np.clip(img_adv, 0, 255)
        else:
            img_adv = img_reshape + perturb*255.0
            img_adv = np.clip(img_adv, 0, 255).astype(np.uint8)
        
        img_adv = img_adv.transpose(1,2,0)
        cv2.imwrite(os.path.join(ad_path, img_save_name), img_adv)


def patch_raodom(source_path, save_path, eps = 2/255,test_img_num = 100,attenuation = False):
    src_path = Path(source_path)
    ad_path = Path(save_path)
    ad_path.mkdir(parents=True, exist_ok=True)
    img_list,txt_list,loc_list = data_loader_test(source_path, True, test_img_num) #The loc_list stores the geometric center points of the ships after the images have been scaled down to 640.
    
    for i in tqdm(range(test_img_num), desc="Processing patch_raodom images", position=0, leave=True):
        txt_path = Path(txt_list[i])
        file_stem = txt_path.stem

        img_file_path = src_path / "images" / f"{file_stem}.jpg"
        img_name = source_path + 'images/' +(txt_list[i].split('/')[-1])[:-3] + 'jpg'
        img_save_name = 'patched_' + (txt_list[i].split('/')[-1])[:-3] + 'jpg'
        
        img_raw = cv2.imread(img_file_path.as_posix())
        img = cv2.imread(img_file_path.as_posix())
        if img_raw is None:
            print(f"Warning: Could not load image {img_name}")
            continue

        img = cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR) #Image scaling interpolation method utilized by yolov5
        img_reshape = img.transpose((2, 0, 1)) #channel,width,height
        image = np.stack([img_reshape], axis=0).astype(np.float32) #convert into batch style

        a = txt_list[i]
        with open(a) as fl:
            img_pre_loc = fl.readlines()
        loc_info_list = [line.split() for line in img_pre_loc]
        ship_loc_list = loc_list[i]
        patch_num = len(ship_loc_list)
        patted_img = np.zeros((640,640,3)).astype(np.uint8)
        x2 = image.copy()

        for k in range(patch_num):
            loc_y = int(256*float(loc_info_list[k][1]))
            loc_x = int(256*float(loc_info_list[k][2]))
            loc_h = int(256*float(loc_info_list[k][3]))
            loc_w = int(256*float(loc_info_list[k][4]))

            original_height, original_width, _ = img_raw.shape
            new_width,new_height,_ = img.shape

            # calculate the proportion 
            width_ratio = new_width / original_width
            height_ratio = new_height / original_height

            # adjust loc_x, loc_y, loc_w, loc_h
            loc_w_scaled = int(loc_w * width_ratio)
            loc_h_scaled = int(loc_h * height_ratio)    

            new_x = int(loc_x * width_ratio - 0.5*loc_w * width_ratio) 
            new_y = int(loc_y * height_ratio - 0.5*loc_h * height_ratio)

            if(new_x > 640 - loc_w_scaled):
                new_x = 640 - loc_w_scaled
            if(new_y > 640 - loc_h_scaled):
                new_y = 640 - loc_h_scaled
            if(new_x < 0):
                new_x = 0
            if(new_y < 0):
                new_y = 0

            current_img = x2.copy()
            # generate perturbation patch from Gaussian distribution
            perturbation = np.random.normal(0, eps, current_img.shape).astype(np.float32)
            perturbation_mask = np.zeros_like(current_img)
            perturbation_mask[:, :, new_x:new_x+loc_w_scaled, new_y:new_y+loc_h_scaled] = 1
            perturbation = perturbation * perturbation_mask

            if attenuation:
                image_adv = current_img - perturbation*255.0
            else:
                image_adv = current_img + perturbation*255.0
            image_adv = np.clip(image_adv, 0, 255).astype(np.uint8)
            x2 = image_adv
            patted_img = x2[0].transpose(1, 2, 0)
        cv2.imwrite(str(ad_path / img_save_name),patted_img)

def background_raodom(source_path, save_path, eps = 2/255,test_img_num = 100,attenuation = False):
    src_path = Path(source_path)
    ad_path = Path(save_path)
    ad_path.mkdir(parents=True, exist_ok=True)
    img_list,txt_list,loc_list = data_loader_test(source_path, True, test_img_num) #The loc_list stores the geometric center points of the ships after the images have been scaled down to 640.
    
    for i in tqdm(range(test_img_num), desc="Processing background_raodom images", position=0, leave=True):
        txt_path = Path(txt_list[i])
        file_stem = txt_path.stem

        img_file_path = src_path / "images" / f"{file_stem}.jpg"
        img_name = source_path + 'images/' +(txt_list[i].split('/')[-1])[:-3] + 'jpg'
        img_save_name = 'patched_' + (txt_list[i].split('/')[-1])[:-3] + 'jpg'
        
        img_raw = cv2.imread(img_file_path.as_posix())
        img = cv2.imread(img_file_path.as_posix())
        if img_raw is None:
            print(f"Warning: Could not load image {img_name}")
            continue

        img = cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR) #Image scaling interpolation method utilized by yolov5
        img_reshape = img.transpose((2, 0, 1)) #channel,width,height
        image = np.stack([img_reshape], axis=0).astype(np.float32) #convert into batch style

        a = txt_list[i]
        with open(a) as fl:
            img_pre_loc = fl.readlines()
        loc_info_list = [line.split() for line in img_pre_loc]
        ship_loc_list = loc_list[i]
        patch_num = len(ship_loc_list)
        patted_img = np.zeros((640,640,3)).astype(np.uint8)
        x2 = image.copy()
        
        # create mask for background area
        background_mask = np.ones_like(x2, dtype=bool)

        for k in range(patch_num):
            loc_y = int(256*float(loc_info_list[k][1]))
            loc_x = int(256*float(loc_info_list[k][2]))
            loc_h = int(256*float(loc_info_list[k][3]))
            loc_w = int(256*float(loc_info_list[k][4]))

            original_height, original_width, _ = img_raw.shape
            new_width,new_height,_ = img.shape

            # calculate the proportion 
            width_ratio = new_width / original_width
            height_ratio = new_height / original_height

            # adjust loc_x, loc_y, loc_w, loc_h
            loc_w_scaled = int(loc_w * width_ratio)
            loc_h_scaled = int(loc_h * height_ratio)    

            new_x = int(loc_x * width_ratio - 0.5*loc_w * width_ratio) 
            new_y = int(loc_y * height_ratio - 0.5*loc_h * height_ratio)

            if(new_x > 640 - loc_w_scaled):
                new_x = 640 - loc_w_scaled
            if(new_y > 640 - loc_h_scaled):
                new_y = 640 - loc_h_scaled
            if(new_x < 0):
                new_x = 0
            if(new_y < 0):
                new_y = 0
            # for single patch
            background_mask[:, :, new_x:new_x+loc_w_scaled, new_y:new_y+loc_h_scaled] = 0

        perturbation = np.random.normal(0, eps, x2.shape).astype(np.float32)
        perturbation = perturbation * background_mask.astype(np.float32) * 255.0

        if attenuation:
            image_adv = x2 - perturbation
        else:
            image_adv = x2 + perturbation

        image_adv = np.clip(image_adv, 0, 255).astype(np.uint8)
        patted_img = image_adv[0].transpose(1, 2, 0)
        cv2.imwrite(str(ad_path / img_save_name), patted_img)


if __name__ == "__main__":
    src_path = './dataset/clean_val'
    save_path = './adv_examples/random_attack28'
    test_img_num = 100
    eps = 28/255


    # for adding part
    whole_random(src_path, os.path.join(save_path, 'whole_adding/'), eps, test_img_num, False)
    patch_raodom(src_path, os.path.join(save_path, 'patch_adding/'), eps, test_img_num, False)
    background_raodom(src_path, os.path.join(save_path, 'back_adding/'), eps, test_img_num, False)

    # for subtracting part
    whole_random(src_path, os.path.join(save_path, 'whole_subtract/'), eps, test_img_num, True)
    patch_raodom(src_path, os.path.join(save_path, 'patch_subtract/'), eps, test_img_num, True)
    background_raodom(src_path, os.path.join(save_path, 'back_subtract/'), eps, test_img_num, True)



  

