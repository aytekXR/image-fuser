from os import listdir
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt


class GlobalVariables:
    _instance = None
    batchSize = None
    epochNum = None
    input_image_dir = None
    output_image_dir = None
    isResume = None
    checkpoint_path = None
    model_path = None
    trainImgList = None
    testsImgList = None
    isRGB = None
    isDeepSupervision = None
    lr = None
    save_model_dir_autoencoder = None
    save_loss_dir = None
    ssim_weight = None
    ssim_path = None
    log_interval = None
    imgHeight = None
    imgWidth = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.batchSize = 2
        self.epochNum = 20
        self.isRGB = False
        self.isResume = True
        self.isDeepSupervision = False
        self.lr = 1e-6
        self.save_model_dir_autoencoder = "models/autoencoder"
        self.save_loss_dir = './models/autoencoder_loss'
        self.ssim_weight = [1,10,100,1000,10000] #convert this to a single value
        self.ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']
        self.log_interval = 10  #"number of images after which the training loss is logged, default is 500"


    def set_vars(self, batchSize=None, epochNum=None, input_image_dir=None, output_image_dir=None, isResume=None, checkpoint_path=None, model_path=None, trainImgList=None, testsImgList=None, isRGB=None):
        if batchSize is not None:
            self.batchSize = batchSize
        if epochNum is not None:
            self.epochNum = epochNum
        if input_image_dir is not None:
            self.input_image_dir = input_image_dir
        if output_image_dir is not None:
            self.output_image_dir = output_image_dir
        if isResume is not None:
            self.isResume = isResume
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        if model_path is not None:
            self.model_path = model_path
        if trainImgList is not None:
            self.trainImgList = trainImgList
        if testsImgList is not None:
            self.testsImgList = testsImgList
        if isRGB is not None:
            self.isRGB = isRGB
        

def list_images(directory):
    images = []
    imgList = listdir(directory)
    imgList.sort()
    images = [filename for filename in imgList if filename.lower().endswith((".png", ".jpg", ".jpeg"))]
    return images

def load_dataset():
    num_images = len(GlobalVariables().trainImgList)
    mod = num_images % GlobalVariables().batchSize
    print('BATCH SIZE: {}'.format(GlobalVariables().batchSize))
    print('Train images number: {}'.format(num_images))
    print('Train images samples: {}'.format(num_images / GlobalVariables().batchSize))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        GlobalVariables().trainImgList = GlobalVariables().trainImgList[:-mod]
    return int(len(GlobalVariables().trainImgList) // GlobalVariables().batchSize)

def get_images_auto(paths):
    images = []
    for path in paths:
        image = get_image(GlobalVariables().input_image_dir + path)
        if GlobalVariables().isRGB is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, GlobalVariables.imgHeight, GlobalVariables.imgWidth])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_image(path):
    if GlobalVariables().isRGB is True:
        image = Image.open(path);
    else:
        image = Image.open(path);
        image = image.convert("L");
    
    if GlobalVariables.imgHeight is None:
        GlobalVariables.imgWidth, GlobalVariables.imgHeight = image.size
    return image.resize((GlobalVariables.imgWidth, GlobalVariables.imgHeight))

def recons_fusion_images(img_lists, h, w):
    img_f_list = []
    h_cen = int(np.floor(h / 2))
    w_cen = int(np.floor(w / 2))
    ones_temp = torch.ones(1, 1, h, w).cuda()
    for i in range(len(img_lists[0])):
        # img1, img2, img3, img4
        img1 = img_lists[0][i]
        savepath = GlobalVariables.output_image_dir + str(i) + ".png"
        save_image_test(img1, savepath)
    return img_f_list

def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    img_fusion = img_fusion.cpu().data[0].numpy()
    
    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    plt.imsave(output_path, img_fusion, cmap='gray')