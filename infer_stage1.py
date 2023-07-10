import argparse
import torch
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import random
from tqdm import tqdm, trange
import os
import time

import net
import utils
import pytorch_msssim


import torchvision.transforms as transforms

def setup(args, globVars):
    imgList = utils.list_images(args.input_image_dir)
    globVars.set_vars(input_image_dir=args.input_image_dir,
                      output_image_dir=args.output_image_dir,
                      checkpoint_path=args.checkpoint_path,
                      model_path=args.model_path,
                      testsImgList=imgList)

def infer(globVars):
    numofChannel = 3 if utils.GlobalVariables().isRGB else 1

    nb_filter = [64, 112, 160, 208, 256]
    nest_model = net.AutoEncoder(nb_filter, numofChannel,
                                 numofChannel, globVars.isDeepSupervision)
    nest_model.load_state_dict(torch.load(globVars.checkpoint_path))

    # print(nest_model)
    nest_model.eval()
    nest_model.cuda()

    print('Start Infering.....')

    if globVars.isRGB :
        print('RGB image is currently not supported!\n')
        # TODO will be coded
    
    for item in globVars.testsImgList:
        images= utils.get_images_auto([item], globVars.testsImgList)
        # images = images[0]
        images = Variable(images, requires_grad=False)
        images = images.cuda()
        en = nest_model.encoder(images)
        out = nest_model.decoder_eval(en)
        savepath = globVars.output_image_dir + "/o" + item
        savepath.replace(".jpg", ".png")
        utils.save_image_test(out[0][0],savepath)

    print("\nDone, Infering. Saved at {}\n".format(globVars.output_image_dir))

def main():
    globVars = utils.GlobalVariables()
    parser = argparse.ArgumentParser(description='Inputs for infer.py file')
    
    # Add arguments
    parser.add_argument('-i', '--input-image-dir' , help='Input Image Directory containing /ir/ and /vis/ folders', default="./tmp/images/")
    parser.add_argument('-c', '--checkpoint-path' , help='Load Model Checkpoints Path', default="./premodels/20230703SSIM_10000/Epoch_19_iters2571_1e5.model")
    parser.add_argument('-o', '--output-image-dir', help='Output Image Directory. Unless given, input dir will be used.', default="./tmp/images/outputs")
    parser.add_argument('-m', '--model-path',       help='Directory containing the models', default="./tmp/models")
    # Parse the arguments
    args = parser.parse_args()

    setup(args,globVars)
    infer(globVars)

if __name__ == '__main__':
    main() 