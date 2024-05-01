import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
from einops import pack, unpack, rearrange
import numpy as np
import glob
import pdb



class Obj3DDataset(Dataset):
    def __init__(self, dataset_path="/mnt/data/shubham/OBJ3D", mode="train") -> None:
        super().__init__()
        # use sorted glob to iterate over train/folder_number 
        # and read in names of all images and store in a list
        folder = os.path.join(dataset_path, mode)
        self.consecutive_frames = [] # t, t+1
        for im_folder in sorted(glob.glob(folder+"/*")):
            images = sorted(glob.glob(im_folder + "/*.png"), 
                            key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
            # create pairs of 3. (t, t+1, t+2) and add them to a global buffer
            for i in range(len(images)-2):
                self.consecutive_frames.append((images[i], images[i+1], images[i+2]))
    
    def __len__(self):
        return len(self.consecutive_frames)
    
    def __getitem__(self, index):
        frames = self.consecutive_frames[index]
        with Image.open(frames[0]) as im:
            frame_t0 = np.array(im)
            frame_t0 = rearrange(frame_t0, "H W C -> C H W")[:3]
        with Image.open(frames[1]) as im:
            frame_t1 = np.array(im)
            frame_t1 = rearrange(frame_t1, "H W C -> C H W")[:3]
        with Image.open(frames[2]) as im:
            frame_t2 = np.array(im)
            frame_t2 = rearrange(frame_t2, "H W C -> C H W")[:3]
        frames, ps = pack([frame_t0[None, ...], 
                            frame_t1[None, ...], 
                            frame_t2[None, ...]],"* C H W")
        # not sure why authors want us to rearrange again
        frames = rearrange(frames, "F C H W -> C F H W")
        return torch.from_numpy(frames)

# train_ds = Obj3DDataset()
# frames = train_ds[0]
# pdb.set_trace()

# simulation has ds like BS, C, 2, H, W