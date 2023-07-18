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
import matplotlib.pyplot as plt

def setup(args, globVars):
    imgListV = utils.list_images(args.input_image_dir+"/vis/")
    imgListI = utils.list_images(args.input_image_dir+"/ir/")
    numofTrainingImages = 8000
    trainImgListV = imgListV[:numofTrainingImages]
    trainImgListI = imgListI[:numofTrainingImages]
    # random.shuffle(trainImgListV)
    # random.shuffle(trainImgListI)
    globVars.set_vars(input_image_dir=args.input_image_dir,
                      output_image_dir=args.output_image_dir,
                      isResume=args.isResume,
                      checkpoint_path=args.checkpoint_path,
                      model_path=args.model_path,
                      trainImgListV=trainImgListV,
                      trainImgListI=trainImgListI)

def train(globVars):
    ssimWeightIndex = -1
    numofChannel = 3 if utils.GlobalVariables().isRGB else 1

    nb_filter = [64, 112, 160, 208, 256]
    nest_model = net.AutoEncoder(nb_filter, numofChannel, numofChannel, globVars.isDeepSupervision)

    if globVars.isResume is True:
        print('Resuming, initializing using weight from {}.'.format(globVars.checkpoint_path))
        nest_model.load_state_dict(torch.load(globVars.checkpoint_path))
    else:
        print('Initializing Weights Randomly')
    # print(nest_model)
    optimizer = Adam(nest_model.parameters(), globVars.lr)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim

    nest_model.cuda()

    tbar = trange(globVars.epochNum)
    print('Start training.....')

    Loss_pixel = []
    Loss_ssim = []
    Loss_all = []
    count_loss = 0
    all_ssim_loss = 0.
    all_pixel_loss = 0.

    for tbarStat in tbar:
        # load training database
        batches = utils.load_dataset()
        nest_model.train()
        count = 0
        for batch in range(batches):
            image_pathsV = globVars.trainImgListV[batch * globVars.batchSize:(batch * globVars.batchSize + globVars.batchSize)]
            image_pathsI = globVars.trainImgListI[batch * globVars.batchSize:(batch * globVars.batchSize + globVars.batchSize)]
            imgV = utils.get_images_auto(image_pathsV, mainfolder=globVars.input_image_dir + "/vis/")
            imgI = utils.get_images_auto(image_pathsI, mainfolder=globVars.input_image_dir + "/ir/")
            
            count += 1
            optimizer.zero_grad()
            imgV = Variable(imgV, requires_grad=False)
            imgV = imgV.cuda()
            imgI = Variable(imgI, requires_grad=False)
            imgI = imgI.cuda()


            # encoder
            enV = nest_model.encoder(imgV)
            enI = nest_model.encoder(imgI)
			
            
            # fuser
            # TODO will be coded
            fused = nest_model.fusion(enI, enV);
            
            # decoder
            outputs = nest_model.decoder_train(fused)

            # resolution loss: between fusion image and visible image
            imgClone = Variable(imgV.data.clone(), requires_grad=False)

            ssim_loss_value = 0.
            pixel_loss_value = 0.
            for output in outputs:
                pixel_loss_temp = mse_loss(output, imgClone)
                ssim_loss_temp = ssim_loss(output, imgClone, normalize=True)
                ssim_loss_value += (1-ssim_loss_temp)
                pixel_loss_value += pixel_loss_temp
            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)

            # total loss
            total_loss = pixel_loss_value + globVars.ssim_weight[ssimWeightIndex] * ssim_loss_value
            # del enV
            # del enI
            # del fused
            # del outputs
            # torch.cuda.empty_cache()
            
            total_loss.backward()
            optimizer.step()
            
            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()

            # print logs
            if (batch + 1) % globVars.log_interval == 0:
                mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.4f}\n".format(
                     time.ctime(), globVars.ssim_weight[ssimWeightIndex], tbarStat + 1, count, batches,
                     all_pixel_loss / globVars.log_interval,
                     (globVars.ssim_weight[ssimWeightIndex] * all_ssim_loss) / globVars.log_interval,
                     (globVars.ssim_weight[ssimWeightIndex] * all_ssim_loss + all_pixel_loss) / globVars.log_interval)
                tbar.set_description(mesg)
                ofile =globVars.save_model_Dir + "/SSIM_{}".format(globVars.ssim_weight[ssimWeightIndex]) + "logfile.txt"
                utils.savelog(ofile, mesg)
                Loss_pixel.append(all_pixel_loss / globVars.log_interval)
                Loss_ssim.append(all_ssim_loss / globVars.log_interval)
                Loss_all.append((globVars.ssim_weight[ssimWeightIndex] * all_ssim_loss + all_pixel_loss) / globVars.log_interval)
                count_loss += 1
                all_ssim_loss = 0.
                all_pixel_loss = 0.

            # Routine Saves
            if((batch+1) % (50*globVars.log_interval)) == 0 or (batches - batch == 1) :
                # Save model
                nest_model.eval()
                nest_model.cpu()
                saveDir = globVars.save_model_Dir + "/SSIM_{}".format(globVars.ssim_weight[ssimWeightIndex])
                if not os.path.exists(saveDir):
                    # Create the directory
                    os.makedirs(saveDir)
                saveDir += "/E{}_i{}.model".format(tbarStat, count)
                torch.save(nest_model.state_dict(), saveDir)

                # TODO: Save loss data 
                # pixel loss
                # SSIM loss
                # all loss
                nest_model.train()
                nest_model.cuda()
                mesg = "\nCheckpoint, trained model saved at {}! \n".format(saveDir)
                tbar.set_description(mesg)
                ofile =globVars.save_model_Dir + "/SSIM_{}".format(globVars.ssim_weight[ssimWeightIndex]) + "logfile.txt"
                utils.savelog(ofile, mesg)

    print("\nDone, trained model saved")
    ofile =globVars.save_model_Dir + "/SSIM_{}".format(globVars.ssim_weight[ssimWeightIndex]) + "logfile.txt"
    utils.savelog(ofile, "\nDone, trained model saved\n\n\n\n\n\n\n")

def main():
    globVars = utils.GlobalVariables()
    parser = argparse.ArgumentParser(description='Inputs for test.py file')
    
    # Add arguments
    parser.add_argument('-i', '--input-image-dir' , help='Input Image Directory containing /ir/ and /vis/ folders', default="/home/ae/repo/03dataset/flir/AnnotatedImages")
    parser.add_argument('-r', '--isResume' , help='set whether contunie training', default= True)
    parser.add_argument('-c', '--checkpoint-path' , help='Load Model Checkpoints Path', default="premodels/fuser/20230716.model")
    parser.add_argument('-o', '--output-image-dir', help='Output Image Directory. Unless given, input dir will be used.', default="/home/ae/repo/03dataset/flir/AnnotatedImages/outputs")
    parser.add_argument('-m', '--model-path',       help='Directory containing the models', default="/home/ae/repo/image-fuser/tmp/models")
    # Parse the arguments
    args = parser.parse_args()

    setup(args,globVars)
    train(globVars)

if __name__ == '__main__':
    main() 