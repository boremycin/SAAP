import numpy as np
import torch
import numpy as np
from torchvision.transforms import transforms
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
import os
import cv2
import sys
sys.path.append('./aap_utils')
from aap_utils.aap_utils import Yolo
from aap_utils.aap_utils import data_loader_test
from aap_utils.adv_attenuation_patch import AAPatch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import yolov5
from tqdm import tqdm
from pathlib import Path
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


# experiment configuration
test_img_num = 1000
save_path = './adv_examples/aap/'
src_path = './dataset/clean_val/'
max_iter = 5000

def generate_adversarial_image_set(source_path,save_path,iterations=300):
    
    attack = AAPatch(
        detector,
        patch_shape=(3, 200, 200),
        patch_location=(50, 50),
        crop_range=(0,0),
        brightness_range=(0.7, 1.4),
        rotation_weights=(0.4, 0.2, 0.2, 0.2),
        sample_size=1,
        learning_rate=5.0,
        max_iter=iterations,
        batch_size=1,
        verbose=True,
        targeted=False,
        energy_tau = 0.5
    ) 
   
    src_path = Path(source_path)
    ad_path = Path(save_path)
    ad_path.mkdir(parents=True, exist_ok=True)
    img_list,txt_list,loc_list = data_loader_test(source_path, True, test_img_num) #The loc_list stores the geometric center points of the ships after the images have been scaled down to 640.
   
    if not os.path.isdir(ad_path):
        os.makedirs(ad_path)

    for i in tqdm(range(test_img_num), desc="Processing images", position=0, leave=True):
        txt_path = Path(txt_list[i])
        file_stem = txt_path.stem

        img_file_path = src_path / "images" / f"{file_stem}.jpg"
        img_save_path = ad_path / f"patched_{file_stem}.jpg"

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

            attack.set_patch_shape((3, loc_w_scaled, loc_h_scaled))
            attack.patch_location = (new_x, new_y)
            
            image_adv = attack.generate(x=x2, y=None)
            image_adv=attack.apply_patch(x=x2,patch_external=image_adv)
           
            x2 = image_adv
            patted_img = x2[0].transpose(1, 2, 0)
        cv2.imwrite(os.path.join(ad_path, img_save_name),patted_img)
generate_adversarial_image_set(src_path,save_path,iterations = max_iter)





