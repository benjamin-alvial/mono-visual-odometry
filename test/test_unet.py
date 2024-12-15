from UNet.unet_model import UNet
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

checkpoint_file = 'UNet/checkpoint_epoch7.pth'
image_path = 'path_to_kitti_image.png'
image_path = "../../data/data_odometry_gray/dataset/sequences/00/image_0/002200.png"

net = UNet(n_channels=3, n_classes=12, bilinear=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)

state_dict = torch.load(checkpoint_file, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)

net.cpu()


print("U-Net loaded correctly")

def preprocess(pil_img, scale=1):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
        img = np.asarray(pil_img)
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))
        if (img > 1).any():
            img = img / 255.0
        return img

img = Image.open(image_path)
img = img.convert('RGB')
img = preprocess(img)
img =  torch.as_tensor(img.copy()).float().contiguous()
img_size = img.size() # 3xHxW
img = img.reshape((1,img_size[0],img_size[1],img_size[2])) # 1x12xHxW

img_pred = net.forward(img) # 1x12xHxW
img_pred = img_pred.detach().numpy()
img_pred = img_pred.squeeze(0) # 12xHxW
img_pred = np.transpose(img_pred, (1, 2, 0))
img_pred = np.argmax(img_pred, 2)

img_original = Image.open(image_path)

print(img_pred)
plt.imshow(img_original)
plt.figure()
plt.imshow(img_pred, vmin=0, vmax=11)
plt.show()


"""
6: vegatation
7: sky
1: road

def kitti_inverse_map_1channel(img):
  cmap = [
    (0, 0), #void (ignorable)
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 1), #road
    (8, 2), #sidewalk
    (9, 2),
    (10, 0), #rail truck (ignorable)
    (11, 3), #construction
    (12, 3),
    (13, 3),
    (14, 3),
    (15, 3),
    (16, 3),
    (17, 4), #pole(s)
    (18, 4),
    (19, 5), #traffic sign
    (20, 5),
    (21, 6), #vegetation
    (22, 6),
    (23, 7), #sky
    (24, 8), #human
    (25, 8),
    (26, 9), #vehicle
    (27, 9),
    (28, 9),
    (29, 9),
    (30, 9),
    (31, 10), #train
    (32, 11), #cycle
    (33, 11)
  ]
"""

