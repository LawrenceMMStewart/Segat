"""
.. module:: segat
   :platform: Python
   :synopsis: An Adversarial Attack module for semantic segmentation neural networks in Pytorch. 
              segat (semantic segmentation attacks) requires that the input images are in the form of the network input.
              For testing a models robustness and adversarial training it advised to avoid the untargeted methods.

.. moduleauthor:: Lawrence Stewart <lawrence.stewart@valeo.com>


"""
from cocoproc import cocodata
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
from pycocotools.coco import COCO
from pycocotools import mask
from cocoproc import custom_transforms
from PIL import Image, ImageFile
from torchvision import transforms as torch_transform
from cocoproc.utils import decode_segmap, get_pascal_labels
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import pylab
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.patches as mpatches
import torch.nn as nn
from skimage.measure import compare_ssim as ssim 

#TO DOS --- Work out what the default clipping value and alpha value should be 
# Implement each of the four attacks


class FGSM():
    """
    FGSM class containing all attacks and miscellaneous functions 
    """

    def __init__(self,model,loss,alpha=1,eps=1): 
        """ Creates an instance of the FGSM class.

        Args:
           model (torch.nn model):  The chosen model to be attacked,
                                    whose output layer returns the nonlinearlity of the pixel
                                    being in each of the possible classes.
           loss  (function):  The loss function to model was trained on
          

        Kwargs:
            alpha (float):  The learning rate of the attack
            eps   (float):  The clipping value for Iterated Attacks

        Returns:
           Instance of the FGSM class
        """
        self.model=model
        self.loss=loss
        self.eps=eps
        self.alpha=alpha
        self.predictor=None
        self.default_its=min(int(self.eps+4),int(1.25*self.eps))
        #set the model to evaluation mode for FGSM attacks
        self.model.eval()


    def untargeted(self,img,pred,labels):
        """Performs a single step untargeted FGSM attack

        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image

        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        l=self.loss(pred,labels)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad=img.grad
        noise=self.alpha*torch.sign(im_grad)
        adv=img+noise
        return adv, noise 



    def targeted(self,img,pred,target):
        """Performs a single step targeted FGSM attack

        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                 for the whole image
            target (torch.tensor):  The target labelling for each pixel


        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        l=self.loss(pred,target)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad=img.grad
        noise=-self.alpha*torch.sign(im_grad)
        adv=img+noise
        return adv, noise



    def iterated(self,img,pred,labels,its=None,targeted=False):
        """Performs iterated untargeted or targeted FGSM attack
        often referred to as FGSMI

        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
                                    or the target labelling we wish the network to misclassify
                                    the network as (this should match the choice of the targeted 
                                    variable)

        Kwargs:
            its (int):  The number of iterations to attack
            targeted (boolean): False for untargeted attack and True for targeted

        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        #set number of iterations to be the default value if not given
        its = self.default_its if its is None else its
        adv=img
        tbar=trange(its)
        for i in tbar:
            pred=self.predictor(adv)
            l=self.loss(pred,labels)
            img.retain_grad()
            torch.sum(l).backward()
            im_grad=img.grad

            #zero the gradients for the next iteration
            self.model.zero_grad()
            #Here the update is GD projected onto ball of radius clipping
            if targeted:
                noise=-self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)
            else:
                noise=self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)

            adv=adv+noise
            tbar.set_description('Iteration: {}/{} of iterated-FGSMI attack'.format(i, its))
        return adv, noise 



    def iterated_least_likely(self,img,pred,its=None):
        """Performs iterated untargeted FGSM attack towards the 
        least likely class, often referred to as FGSMII

        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image

        Kwargs:
            its (int):  The number of iterations to attack 

        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        #set number of iterations to be the default value if not given
        its = self.default_its if its is None else its
        adv=img
        with torch.no_grad():
            pred=self.predictor(adv)
            targets=torch.argmax(pred[0],0)
            targets=targets.reshape(1,targets.size()[0],-1)
        tbar = trange(its)
        for i in tbar:
            pred=self.predictor(adv)
            l=self.loss(pred,targets)
            img.retain_grad()
            torch.sum(l).backward()
            im_grad=img.grad

            #zero the gradients for the next iteration
            self.model.zero_grad()
            #Here the update is GD projected onto ball of radius clipping
            noise=-self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)
            adv=adv+noise
            tbar.set_description('Iteration: {}/{} of iterated-FGSMII attack'.format(i, its))
        return adv, noise 




    def ssim_iterated(self,img,pred,labels,its=None,targeted=False,threshold=0.99):
        """Performs iterated untargeted or targeted FGSM attack
        often referred to as FGSMI, halfing the current value of 
        alpha until the ssim value between the origional and perturbed image
        reaches the threshold value

        Args:
            model (torch.nn model): The pytorch model to be attacked
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image
                                    or the target labelling we wish the network to misclassify
                                    the network as (this should match the choice of the targeted 
                                    variable)

        Kwargs:
            its (int):  The number of iterations to attack
            targeted (boolean): False for untargeted attack and True for targeted
            threshold (float): Threshold ssi value to obtain

        Returns:
           adv (torch.tensor):  The pertubed image
           noise (torch.tensor): The adversarial noise added to the image during the attack
        """

        #set number of iterations to be the default value if not given


        ssim_val=0
        counter=0
        self.alpha=self.alpha*2
        its = self.default_its if its is None else its

        tbar=trange(its)
        
        img_ar=img[0].cpu().detach().numpy()
        img_ar=np.transpose(img_ar,(1,2,0))
        img_ar=self.denormalise(img_ar)

        while ssim_val<0.99:
            self.alpha = self.alpha/2
            adv=img
            counter+=1
            for i in tbar:
                pred=self.predictor(adv)
                l=self.loss(pred,labels)
                img.retain_grad()
                torch.sum(l).backward()
                im_grad=img.grad

                #zero the gradients for the next iteration
                self.model.zero_grad()
                #Here the update is GD projected onto ball of radius clipping
                if targeted:
                    noise=-self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)
                else:
                    noise=self.alpha*torch.sign(im_grad).clamp(-self.eps,self.eps)

                adv=adv+noise
                tbar.set_description('Iteration: {}/{} of iterated-FGSMI attack- attempt {}'.format(i, its,counter))

            # convert to numpy array
            adv_ar=adv[0].cpu().detach().numpy()
            adv_ar=np.transpose(adv_ar,(1,2,0))
            adv_ar=self.denormalise(adv_ar)

            ssim_val=ssim(adv_ar,img_ar,multichannel=True)


        return adv, noise 




    def untargeted_varied_size(self,img,pred,labels,alphas=[0,0.005,0.01]):
        """Performs a single step untargeted FGSM attack for each of the given
        values of alpha.

        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            labels (torch.tensor):  The true labelelling of each pixel in the image

        Kwargs:
            alphas (float list): The values of alpha to perform the attacks with

        Returns:
           adv_list (torch.tensor list):  The list of the pertubed images
           noise_list (torch.tensor list): The list of the adversarial noises created
        """
        if alphas==[]:
            raise Exception("alphas must be a non empty list")
        #create the output lists
        l=self.loss(pred,labels)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad=img.grad
        noise_list=[alpha*torch.sign(im_grad) for alpha in alphas]
        adv_list=[img+noise for noise in noise_list]
        return adv_list, noise_list 
        

    def targeted_varied_size(self,img,pred,target,alphas=[0,0.005,0.01]):
        """Performs a single step targeted FGSM attack for each of the given
        values of alpha.

        Args:
            img (torch.tensor):  The image to be pertubed to attack the network
            pred (torch.tensor): The prediction of the network for each pixel 
                                for the whole image
            target (torch.tensor):  The target labelling that we wish the network
                                    to mixclassify the pixel as.

        Kwargs:
            alphas (float list): The values of alpha to perform the attacks with

        Returns:
           adv_list (torch.tensor list):  The list of the pertubed images
           noise_list (torch.tensor list): The list of the adversarial noises created
        """
        if alphas==[]:
            raise Exception("alphas must be a non empty list")
        #create the output lists
        l=self.loss(pred,target)
        img.retain_grad()
        torch.sum(l).backward()
        im_grad=img.grad
        noise_list=[-alpha*torch.sign(im_grad) for alpha in alphas]
        adv_list=[img+noise for noise in noise_list]
        
        return adv_list, noise_list 



    def DL3_pred(self,img):
        """Extractor function for deeplabv3 pretained: Please add your own
            to the self.predictor variable to suite your networks output 

        Args:
            img (torch.tensor):  The image to be pertubed to attack the network

        Returns: out (torch.tensor): Predicted semantic segmentation
           
        """
        out=self.model(img)['out']
        return out


    def denormalise(self,img):
        """
        Denormalises an image using the image net mean and 
        std deviation values.

        Args:
            img (numpy array): The image to be denormalised

        Returns:
            img (numpy array): The denormalised image
        """
        img *= (0.229, 0.224, 0.225)
        img += (0.485, 0.456, 0.406)
        img *= 255.000
        img=img.astype(np.uint8).clip(0,255)
        return img




#Create a demo adversarial attack:

if __name__ == "__main__":
    #options are cpu or cuda
    device = torch.device('cuda')   

    model = deeplabv3_resnet101(pretrained=True)
    model.to(device)
    model.eval()
    loss=nn.CrossEntropyLoss(reduction="none")

    #arguements for creating the cocosegmentation class
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513


    processing = cocodata.COCOSegmentation(args, split='val', year='2017')
    dataloader = DataLoader(processing, batch_size=1, shuffle=True, num_workers=0)


    for ii, sample in enumerate(dataloader):

        if ii == 1:
            break

        img = sample['image'].numpy()
        gt = sample['label'].numpy()

        #converts to integer
        tmp = np.array(gt[0]).astype(np.uint8) 

        #decode the segmentation map for plotting
        segmap = decode_segmap(tmp, dataset='coco')
        img_tmp = np.transpose(img[0], axes=[2, 0, 1])# img_tmp has dim hxwx3


        #Transpose to the ordering of dimensions expected by Deeplab
        rimg_tmp=np.transpose(img_tmp,axes=[2,0,1]) #has dim 3xhxw
        rimg_tmp=np.array([rimg_tmp])
        img_tens=torch.tensor(rimg_tmp,requires_grad=True).type(torch.FloatTensor).to(device)

        print("predicting segementation using DeeplabV3-Resnet 101 backbone",end="\n",flush=True)
        # predictedraw=model(torch.tensor(img))
        model_out=model(img_tens)
        print("Generating adversarial example",end="\n",flush=True)

        #instantiate the fgsm class 
        attack=FGSM(model,loss)
        attack.predictor=attack.DL3_pred


        #predict segmentation for the image
        #the conv output is not normalised between 0 and 1 but is in the form of non-normalised class probabilites
        conv_output=attack.DL3_pred(img_tens)
        targets=torch.tensor([tmp]).type(torch.LongTensor).to(device)

        single=False
        varied= not single

        #--------------------------------Options: Single Attack-------------------------

        if single:


            #Run the adversarial attack (targets are the labels)
            adv_img_tens, _ =attack.iterated_least_likely(img_tens,conv_output)

            #extract to numpy array and convert back to hxwx3
            adv_img_tmp=adv_img_tens[0].cpu().detach().numpy()
            adv_img_tmp=np.transpose(adv_img_tmp,axes=[1,2,0])


            #predict the adversarially attacked image
            with torch.no_grad():
                adv_out=attack.DL3_pred(adv_img_tens)
            adv_pred=torch.argmax(adv_out[0],0).cpu().detach().numpy()
            adv_pred=np.array([adv_pred])
            adv_tmp=np.array(adv_pred[0]).astype(np.uint8)
            adv_decoded=decode_segmap(adv_tmp,dataset="coco")

            #Unormalise both the images
            img_tmp=processing.denormalise(img_tmp)
            adv_img_tmp=processing.denormalise(adv_img_tmp)
            #compute mIOU
            miou=processing.mIOU(segmap,adv_decoded)



    #--------------------------------Options: Varied Attacks -------------------------
        if varied:
            #l_ represents the list version of the variable above
            #Run the adversarial attack (targets are the labels)
            alphas=[0,0.005,0.01]
            l_adv_img_tens, _ =attack.untargeted_varied_size(img_tens,conv_output,targets,alphas)
            #extract to numpy array and convert back to hxwx3
            ladv_img_tmp=[adv_img_tens[0].cpu().detach().numpy() for adv_img_tens in l_adv_img_tens]
            ladv_img_tmp=[np.transpose(adv_img_tmp,axes=[1,2,0]) for adv_img_tmp in ladv_img_tmp]


            #predict the adversarially attacked image
            with torch.no_grad():
                l_adv_out=[attack.DL3_pred(adv_img_tens) for adv_img_tens in l_adv_img_tens]
            l_adv_pred=[torch.argmax(adv_out[0],0).cpu().detach().numpy() for adv_out in l_adv_out]
            l_adv_pred=[np.array([adv_pred]) for adv_pred in l_adv_pred]
            l_adv_tmp=[np.array(adv_pred[0]).astype(np.uint8) for adv_pred in l_adv_pred]
            l_adv_decoded=[decode_segmap(adv_tmp,dataset="coco") for adv_tmp in l_adv_tmp]

            #Unormalise both the images
            img_tmp=processing.denormalise(img_tmp)
            l_adv_img_tmp=[processing.denormalise(adv_img_tmp) for adv_img_tmp in ladv_img_tmp]
            #create a list of segmap so it can be zipped when calculating the mIOU for each attack level
            l_tmp=[tmp for i in range(len(l_adv_tmp))]
            l_miou=[processing.mIOU(g_truth,n_pred, ) for g_truth, n_pred in zip(l_tmp,l_adv_tmp)]



        #Process predicted segmentation mask ---- 
        pred=torch.argmax(conv_output[0],0).cpu().detach().numpy()
        pred=np.array([pred])
        pred_tmp = np.array(pred[0]).astype(np.uint8) 
        pred_decoded=decode_segmap(pred_tmp,dataset="coco")






    #Clear the saved image repository and then plot the segmentations       
    processing.clear_saved_plots()

    if single:
        processing.plot_seg(img_tmp,pred_decoded,"Deeplabv3")
        processing.plot_seg(img_tmp,segmap,"Groundtruth")
        processing.plot_seg(adv_img_tmp,adv_decoded,"adversarial")

        processing.save_adv(img_tmp,adv_img_tmp,pred_decoded,adv_decoded)


        print("Segmentation Accuracy vanilla pred",(pred_decoded==segmap).sum()/segmap.size)
        print("Segmentation Accuracy adversarial",(adv_decoded==segmap).sum()/segmap.size)

    if varied:
        print("The errors are ",l_miou)
        processing.plot_3x2(l_adv_img_tmp,l_adv_decoded,alphas,errors=l_miou)


