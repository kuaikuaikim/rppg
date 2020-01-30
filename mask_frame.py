import cv2
import numpy as np
import torch
from torch import nn
from models import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageFilter
import time
import sys


class MaskFrame():

    def __init__(self, bs):
        self.frame_counter = 0
        self.batch_size = bs
        self.stop = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        self.model.load_state_dict(torch.load('linknet.pth', map_location='cuda:0'))
        self.model.eval()
        self.model.to(self.device)


    def get_mask_frame(self, orig):

        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model.eval()

        shape = orig.shape[0:2]
        frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256), cv2.INTER_LINEAR)

        a = img_transform(Image.fromarray(frame))
        a = a.unsqueeze(0)
        imgs = Variable(a.to(dtype=torch.float, device=self.device))
        pred = self.model(imgs)

        pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
        mask = pred.data.cpu().numpy()
        mask = mask.squeeze()

        # im = Image.fromarray(mask)
        # im2 = im.filter(ImageFilter.MinFilter(3))
        # im3 = im2.filter(ImageFilter.MaxFilter(5))
        # mask = np.array(im3)

        mask = mask > 0.8
        orig[mask == 0] = 0
        return [orig]



