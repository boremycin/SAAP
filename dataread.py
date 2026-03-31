from torch.utils.data import Dataset
import os
 
class MyData(Dataset):
    def __init__(self, root_dir, image_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.image_list = os.listdir(self.image_path)
        self.image_list.sort()
 
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)
        img_item_path.replace("\\", "/")
        return img_item_path
    def __len__(self):
        return len(self.image_list)
 
if __name__ == '__main__':
    root_dir = ""
    image_ants = "adversarial-1"
    ants_dataset = MyData(root_dir, image_ants)
    print(ants_dataset[1])
 
 
 