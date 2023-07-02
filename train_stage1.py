import argparse
import torch
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import random
from tqdm import tqdm, trange
import os

import net
import utils
import pytorch_msssim

def setup(args):
    globVars = utils.GlobalVariables()
    imgList = utils.list_images(args.input_image_dir)
    numofTrainingImages = 8000
    trainImgList = imgList[:numofTrainingImages]
    random.shuffle(trainImgList)
    globVars.set_vars(input_image_dir=args.input_image_dir, output_image_dir=args.output_image_dir, isResume=args.isResume, checkpoint_path=args.checkpoint_path, model_path=args.model_path, trainImgList=trainImgList)

def train():
    globVars = utils.GlobalVariables()
    ssimWeightIndex = -1
    numofChannel = 3 if utils.GlobalVariables().isRGB else 1

    nb_filter = [64, 112, 160, 208, 256]
    nest_model = net.AutoEncoder(nb_filter, numofChannel, numofChannel, globVars.isDeepSupervision)

    if globVars.isResume is True:
        print('Resuming, initializing using weight from {}.'.format(globVars.checkpoint_path))
        nest_model.load_state_dict(torch.load(globVars.checkpoint_path))
    else:
        print('Initializing Weights Randomly')
    print(nest_model)
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
            image_paths = globVars.trainImgList[batch * globVars.batchSize:(batch * globVars.batchSize + globVars.batchSize)]
            img = utils.get_train_images_auto(image_paths)
            count += 1
            optimizer.zero_grad()
            img = Variable(img, requires_grad=False)
            img = img.cuda()
            
            # encoder
            en = nest_model.encoder(img)
			# decoder
            outputs = nest_model.decoder_train(en)

            # resolution loss: between fusion image and visible image
            imgClone = Variable(img.data.clone(), requires_grad=False)

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
            total_loss.backward()
            optimizer.step()
            
            all_ssim_loss += ssim_loss_value.item()
            all_pixel_loss += pixel_loss_value.item()

            # print logs
            if (batch + 1) % globVars.log_interval == 0:
                mesg = "{}\t SSIM weight {}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t total: {:.6f}".format(
                     time.ctime(), globVars.ssim_weight[ssimWeightIndex], tbarStat + 1, count, batches,
                     all_pixel_loss / globVars.log_interval,
                     (globVars.ssim_weight[ssimWeightIndex] * all_ssim_loss) / globVars.log_interval,
                     (globVars.ssim_weight[ssimWeightIndex] * all_ssim_loss + all_pixel_loss) / globVars.log_interval)
                tbar.set_description(mesg)
                Loss_pixel.append(all_pixel_loss / globVars.log_interval)
                Loss_ssim.append(all_ssim_loss / globVars.log_interval)
                Loss_all.append((globVars.ssim_weight[ssimWeightIndex] * all_ssim_loss + all_pixel_loss) / globVars.log_interval)
                count_loss += 1
                all_ssim_loss = 0.
                all_pixel_loss = 0.

            # Routine Saves
            if((batch+1) % (200*globVars.log_interval)) == 0 or (batches - batch == 1) :
                # Save model
                nest_model.eval()
                nest_model.cpu()
                saveDir = globVars.save_model_dir_autoencoder + "/SSIM_{}".format(globVars.ssim_weight[ssimWeightIndex])
                if not os.path.exists(saveDir):
                    # Create the directory
                    os.makedirs(saveDir)
                saveDir += "/Epoch_{}_iters{}.model".format(tbarStat, count)
                torch.save(nest_model.state_dict(), saveDir)

                # TODO: Save loss data 
                # pixel loss
                # SSIM loss
                # all loss
                nest_model.train()
                nest_model.cuda()
                tbar.set_description("\nCheckpoint, trained model saved at {}".format(saveDir))

    print("\nDone, trained model saved")

def main():
    parser = argparse.ArgumentParser(description='Inputs for test.py file')
    
    # Add arguments
    parser.add_argument('-i', '--input-image-dir' , help='Input Image Directory containing /ir/ and /vis/ folders', default="/home/ae/repo/02paper-repos/mycode/image-fuser/tmp/images/")
    parser.add_argument('-r', '--isResume' , help='set whether contunie training', default= False)
    parser.add_argument('-c', '--checkpoint-path' , help='Load Model Checkpoints Path', default="/home/ae/repo/image-fuser/tmp/models")
    parser.add_argument('-o', '--output-image-dir', help='Output Image Directory. Unless given, input dir will be used.', default="/home/ae/repo/image-fuser/tmp/images/outputs")
    parser.add_argument('-m', '--model-path',       help='Directory containing the models', default="/home/ae/repo/image-fuser/tmp/models")
    # Parse the arguments
    args = parser.parse_args()

    setup(args)
    train()

if __name__ == '__main__':
    main() 