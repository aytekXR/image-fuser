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
    imgListV = utils.list_images(args.input_image_dir+"/vis/")
    imgListI = utils.list_images(args.input_image_dir+"/ir/")

    globVars.set_vars(input_image_dir=args.input_image_dir,
                      output_image_dir=args.output_image_dir,
                      checkpoint_path=args.checkpoint_path,
                      model_path=args.model_path,
                      testsImgListV=imgListV,
                      testsImgListI=imgListI)

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
    
    for item1, item2 in zip(globVars.testsImgListV, globVars.testsImgListI):
        imagesV= utils.get_images_auto([item1], globVars.input_image_dir +'/vis/')
        imagesI= utils.get_images_auto([item2], globVars.input_image_dir +'/ir/')
        # images = images[0]
        imagesV = Variable(imagesV, requires_grad=False)
        imagesI = Variable(imagesI, requires_grad=False)
        imagesV = imagesV.cuda()
        imagesI = imagesI.cuda()
        
        # encoder
        enV = nest_model.encoder(imagesV)
        enI = nest_model.encoder(imagesI)
        # fuser
        fused = nest_model.fusion(enV, enI);
        # decoder
        out = nest_model.decoder_eval(fused)
        
        savepath = globVars.output_image_dir + "/o" + item1
        savepath.replace(".jpg", ".png")
        utils.save_image_test(out[0][0],savepath)

    print("\nDone, Infering. Saved at {}\n".format(globVars.output_image_dir))

def main():
    globVars = utils.GlobalVariables()
    parser = argparse.ArgumentParser(description='Inputs for infer.py file')
    
    # Add arguments
    parser.add_argument('-i', '--input-image-dir' , help='Input Image Directory containing /ir/ and /vis/ folders', default="tmp/single")
    parser.add_argument('-c', '--checkpoint-path' , help='Load Model Checkpoints Path', default="/home/ae/repo/image-fuser/premodels/fuser/Epoch_3_iters5142.model")
    parser.add_argument('-o', '--output-image-dir', help='Output Image Directory. Unless given, input dir will be used.', default="tmp/single/outputs")
    parser.add_argument('-m', '--model-path',       help='Directory to save the models', default="/home/ae/repo/image-fuser/tmp/models")
    # Parse the arguments
    args = parser.parse_args()

    setup(args,globVars)
    infer(globVars)

if __name__ == '__main__':
    main() 