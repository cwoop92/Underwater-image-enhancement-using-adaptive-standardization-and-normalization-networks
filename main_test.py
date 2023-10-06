import torch
import torchvision
import glob
import os
import model
import numpy as np
from PIL import Image

def Img_Loader(image_path):

    img_haze = Image.open(image_path)
    img_haze = (np.asarray(img_haze)/255.0)
    img_haze = torch.from_numpy(img_haze).float()
    img_haze = img_haze.permute(2,0,1)
    img_haze = img_haze.cuda().unsqueeze(0)

    return img_haze

def dehaze_image(image_path):

    Transmission_Net = model.Transmission_Net().cuda()

    test_path = 'test/'
    Transmission_Net.load_state_dict(torch.load('snapshots\\dehazer_optimize_final_confirm.pth'))

    if not os.path.exists(test_path):
        os.mkdir(test_path)

    img_haze = Img_Loader(image_path)
    air_map = Transmission_Net(img_haze)
    torchvision.utils.save_image(air_map, test_path  + "\\" + image_path.split("\\")[-1])

if __name__ == '__main__':

    test_list = glob.glob("samples\\*")

    for image in test_list:

        dehaze_image(image)
        print(image, "done!")


