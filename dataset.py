from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as TF


class DataLoaderTurb(Dataset):
    def __init__(self, turb_dir, num_frames=10, noise=0):
        super(DataLoaderTurb, self).__init__()
        self.num_frames = num_frames
        self.img_list = [os.path.join(turb_dir, d) for d in os.listdir(turb_dir)]
        self.sizex = len(self.img_list)  # get the size of target
        self.noise = noise

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        frame = Image.open(self.img_list[index_]).convert("RGB")
        return TF.to_tensor(frame)
