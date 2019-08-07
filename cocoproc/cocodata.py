#Preprocessing for the subset of Coco that deeplab v3 is trained on (the set containing pascal VOC labels).
#Note this requires the  pycocotools package available on github

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
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable



class COCOSegmentation(Dataset):

    NUM_CLASSES = 21
    #these are the classes that deeplabv3 was pretrained on
    label_dict={1:"Person", 3:"Car", 2:"Bicycle", 6:"Bus", 4:"Motorbike", 7:"Train", 5:"aeroplane",
                62:"Chair", 44:"Bottle", 67:"Dining Table", 64:"Potted Plant", 72:"TV monitor", 9:"Boat",
                63:"Sofa", 16:"Bird", 17:"Cat", 21:"Cow", 18:"Dog", 19:"Horse", 20:"Sheep", 0:"Background"
    }
    # The list of the categories w.r.t. the COCO labelling system that appear in the training set of Deelabv3. Please note that
    #the order is neccessary otherwise the classes function is not alligned correctly.
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
            1, 64, 20, 63, 7, 72]


    def __init__(self,
                 args,
                 base_dir='/datasets_master/COCO',
                 split='val',
                 year='2017'):
        super().__init__()

        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        #--> '/datasets_master/COCO/annotations/instances_train2017.json'
        ids_file = os.path.join('/datasets_local/COCO/', 'annotations/{}_ids_{}.pth'.format(split, year))
        #--> '/datasets_local/COCO/annotations/train_ids_2017.pth'
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year)) #sets image directory to train2017 with path
        self.split = split
        # initialize COCO api for instance annotations
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys()) #coco has a dictionary for each image so we are getting the keys
            self.ids = self._preprocess(ids, ids_file)

        self.args = args




    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        #--------------- Normalise Images for Deeplabv3 ------------------------
        if self.split == "train":
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
         #----------------------------------------------------------------------
        
        sample = self.transform_weak(sample)
        sample['image_name'] = str(self.ids[index]).zfill(12)
        return sample



    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
        _target = Image.fromarray(_mask)
        return _img, _target



    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, please note this may take a while " + \
              "-- it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])

            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)

            tbar.set_description('Iteration: {}/{}, found {} qualifying images'.format(i, len(ids), len(new_ids)))


        print('Number of qualified images found: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids


    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for i, instance in enumerate(target):
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask




    #train set transformation using the imagenet values
    def transform_tr(self, sample):
        composed_transforms = torch_transform.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            custom_transforms.RandomGaussianBlur(),
            custom_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            custom_transforms.ToTensor()])

        return composed_transforms(sample)

    #a transformation which just changes the format to tensor
    def transform_weak(self, sample):
        composed_transforms = torch_transform.Compose([
            custom_transforms.ToTensor()])

        return composed_transforms(sample)

    #validation set transormation using imagenet values
    def transform_val(self, sample):

        composed_transforms = torch_transform.Compose([
            custom_transforms.FixScaleCrop(crop_size=self.args.crop_size),
            custom_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            custom_transforms.ToTensor()])

        return composed_transforms(sample)


    def __len__(self):
        return len(self.ids)


    def denormalise(self,img):
        """
        Denormalises an image using the image net mean and 
        std deviation values.

        Args:
            img (numpy array): The image to be denormalised having shape
            hxwx3

        Returns:
            img (numpy array): The denormalised image
        """
        img  *=(0.229, 0.224, 0.225)
        img  +=(0.485, 0.456, 0.406)
        img   *= 255.000
        img=img.astype(np.uint8)
        img=img.clip(0,255)
        return img


    #rgb to label is used for creating colormaps for the segmenations
    def rgb_to_label_dict(self):
        #creates a dictionary mapping a rbg value (in string form) to a label name
        labels=self.classes
        #create a square matrix consisting of numbers from 0-20 in order to decode and get rgb basis 
        basis=np.array([[i]*self.NUM_CLASSES for i in range(self.NUM_CLASSES)])
        #decode
        de_basis=decode_segmap(basis, dataset="coco")
        #take the rgb value of each of the classes and create a list containing all
        de_basis_flat=[str(de_basis.tolist()[i][0]) for i in range(de_basis.shape[0])]
        #create the dictionary 
        return dict(zip(de_basis_flat,labels))


    # def mIOU(self,truth,prediction):
    #     conf=confusion_matrix(truth.flatten(),prediction.flatten(),[i for i in range(21)])
    #     intersection=np.diag(conf)
    #     union=np.sum(conf,axis=0)
        
    #     #find the present classes:
    #     pres_classes=np.nonzero(intersection)[0]
    #     present_int=intersection[pres_classes]
    #     present_union=union[pres_classes]
    #     #find all present classes and return their mean
    #     return np.mean(present_int/present_union) 
    def fast_hist(self,a, b):
        n=self.NUM_CLASSES
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def mIOU(self,truth,prediction):
        hist=self.fast_hist(truth,prediction)
        return np.nanmean(self.per_class_iu(hist))



    


    def plot_seg(self,image,seg,title="imageseg",save=True,overlap=True,miou=None):
        #plots the image and its segmenation (decoded) with title with the options 
        #to save the image and having the segmentation over the image
        ref_dict=self.rgb_to_label_dict()
        plt.figure()
        plt.title(title)
        plt.subplot(211)
        plt.axis("off")
        plt.imshow(image)
        plt.subplot(212)
        if overlap:
            plt.imshow(image)
        im= plt.imshow(seg,alpha=0.8)
        plt.axis("off")

        if miou!=None:
            plt.title("mIOU = %.3f"%miou)

        #obtain all unique rgbs that the image has:
        reshaped_seg=seg.reshape(-1,seg.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.show(block=True)
        if save:
            pylab.savefig("./saved_plots/"+title+".png")

    def save_adv(self,image,adv_image,seg,adv_seg,title="segmentations",save=True,overlap=True):
        #plots the image and adversarial image next to each other and then also plots 
        #the models segmentation and adversarial segmentation below next to each other
        ref_dict=self.rgb_to_label_dict()
        plt.figure(figsize=(14,7))
        plt.subplot(221)
        plt.title("Image")
        plt.imshow(image)
        plt.subplot(222)
        plt.title("Adversarial Image")
        plt.imshow(adv_image)
        plt.subplot(223)
        plt.title("Image Segmentation")
        if overlap:
            plt.imshow(image)
        im= plt.imshow(seg,alpha=0.8)
        #obtain all unique rgbs that the image has:
        reshaped_seg=seg.reshape(-1,seg.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

        plt.subplot(224)
        plt.title("Adversarial Image Segmentation")
        if overlap:
            plt.imshow(adv_image)
        im= plt.imshow(adv_seg,alpha=0.8)
        plt.axis("off")
        #obtain all unique rgbs that the image has:
        reshaped_seg=adv_seg.reshape(-1,adv_seg.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.show(block=True)
        if save:
            pylab.savefig("./saved_plots/"+title+".png")


    def plot_3x2(self,imagelist,seglist,alphas=[0,0.005,0.01],title="varied_step_sizes",save=True,errors=None):
        if len(alphas)!=3 or len(seglist)!=3 or len(imagelist)!=3:
            raise Exception("Expects 3 images,segmenations and alphas values in list form")
        ref_dict=self.rgb_to_label_dict()
        fig, axarray= plt.subplots(len(seglist),2)
        for i in range(3):
            seg=seglist[i]
            img=imagelist[i]
            axarray[i,0].imshow(img)
            axarray[i,1].imshow(img)
            axarray[i,0].axis("off")
            axarray[i,1].axis("off")
            #create plot titles
            alpha=alphas[i]
            if errors!=None:
                e=errors[i]
                axarray[i,1].set_title(r"Segmentation $\alpha=%.3f$, mIOU=%.3f"%(alpha,e))
            else:
                axarray[i,1].set_title(r"Segmentation $\alpha=%.3f$"%alpha)
            axarray[i,0].set_title(r"Image $\alpha=%.3f$"%alpha)
            im=axarray[i,1].imshow(seg,alpha=0.8)
            #obtain all unique rgbs that the image has:
            reshaped_seg=seg.reshape(-1,seg.shape[-1]).tolist()
            #unique values
            values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
            # create a patch (proxy artist) for every color 
            patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
            # put those patched as legend-handles into the legend
            axarray[i,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.tight_layout()
        plt.show(block=True)
        if save:
            pylab.savefig("./saved_plots/"+title+".png")


    def plot_targeted(self,o_im,t_im,o_s,t_s,adv_im,adv_pred,title="Targeted_FGSM",save=True,mIOUs=None):
        ref_dict=self.rgb_to_label_dict()
        fig,axarray=plt.subplots(2,3,figsize=(14,6))
        #turn of the axis:
        for i in range(2):
            for j in range(3):
                axarray[i,j].axis("off")

        #plot origional image and its predicted segmentation map
        axarray[0,0].imshow(o_im)
        axarray[0,0].set_title("Input Image")
        axarray[1,0].imshow(o_im)
        im=axarray[1,0].imshow(o_s,alpha=0.8)
        #create the legend
        reshaped_seg=o_s.reshape(-1,o_s.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        axarray[1,0].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        if mIOUs==None:
            axarray[1,0].set_title("DLV3 Prediction")
        else:
            axarray[1,0].set_title("mIOU = %.3f"%(mIOUs[0]))

        #plot target image and segmentation map
        axarray[0,1].imshow(t_im)
        axarray[0,1].set_title("Target Image")
        axarray[1,1].imshow(t_im)
        im=axarray[1,1].imshow(t_s,alpha=0.8)
        #create the legend
        reshaped_seg=t_s.reshape(-1,t_s.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        axarray[1,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        axarray[1,1].set_title("Ground Truth")


        #plot the adversarial image and its predicted segmentation map
        axarray[0,2].imshow(adv_im)
        axarray[0,2].set_title("Adversarial Image")
        axarray[1,2].imshow(adv_im)
        im=axarray[1,2].imshow(adv_pred,alpha=0.8)
        #create the legend
        reshaped_seg=adv_pred.reshape(-1,adv_pred.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        axarray[1,2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        
        if mIOUs==None:
            axarray[1,2].set_title("DLV3 Adversarial Prediction")
        else:
            axarray[1,2].set_title("mIOU = %.3f"%(mIOUs[1]))

        plt.tight_layout()
        plt.show(block=True)
        if save:
            pylab.savefig("./saved_plots/"+title+".png")









    def plot_noises(self,advs,noises,segs,alphas,mious=None,title="noises"):

        #assuming the input is correct obtain the number of images to plot
        #in our example this will be 3.
        n=len(alphas)
        ref_dict=self.rgb_to_label_dict()
        fig,axarray=plt.subplots(3,n,figsize=(14,6))
        #turn of the axis 
        for i in range(3):
            for j in range(n):
                axarray[i,j].axis("off")

        #plot the images:
        for j in range(n):

            #plot the adversarial image:
            axarray[0,j].imshow(advs[j])
            axarray[0,j].set_title(r"Adversarial Image $\alpha = %.3f$"%(alphas[j]))
            #plot the noise
            axarray[1,j].imshow(noises[j])
            axarray[1,j].set_title(r"Perturbation $\alpha = %.3f$"%(alphas[j]))

            #plot the segmentation
            axarray[2,j].imshow(advs[j])
            im=axarray[2,j].imshow(segs[j],alpha=0.8)
            reshaped_seg=segs[j].reshape(-1,segs[j].shape[-1]).tolist()
            values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
            patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
            axarray[2,j].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
            #titles
            if mious==None:
                axarray[2,j].set_title("Predicted Segmentation")
            else:
                axarray[2,j].set_title("mIOU = %.3f"%(mious[j]))

        plt.tight_layout()
        plt.show(block=True)
        pylab.savefig("./saved_plots/"+title+".png")








    def plot_targeted_ssim(self,o_im,t_im,o_s,t_s,adv_im,adv_pred,ssim,title="Targeted_FGSM",save=True,mIOUs=None):
        ref_dict=self.rgb_to_label_dict()
        fig,axarray=plt.subplots(2,3,figsize=(14,6))
        #turn of the axis:
        for i in range(2):
            for j in range(3):
                axarray[i,j].axis("off")

        #plot origional image and its predicted segmentation map
        axarray[0,0].imshow(o_im)
        axarray[0,0].set_title("Input Image")
        axarray[1,0].imshow(o_im)
        im=axarray[1,0].imshow(o_s,alpha=0.8)
        #create the legend
        reshaped_seg=o_s.reshape(-1,o_s.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        axarray[1,0].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        if mIOUs==None:
            axarray[1,0].set_title("DLV3 Prediction")
        else:
            axarray[1,0].set_title("mIOU = %.3f"%(mIOUs[0]))

        #plot target image and segmentation map
        axarray[0,1].imshow(t_im)
        axarray[0,1].set_title("Target Image")
        axarray[1,1].imshow(t_im)
        im=axarray[1,1].imshow(t_s,alpha=0.8)
        #create the legend
        reshaped_seg=t_s.reshape(-1,t_s.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        axarray[1,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        axarray[1,1].set_title("Ground Truth")


        #plot the adversarial image and its predicted segmentation map
        axarray[0,2].imshow(adv_im)
        axarray[0,2].set_title("Adversarial SSIM = %.3f"%(ssim))
        axarray[1,2].imshow(adv_im)
        im=axarray[1,2].imshow(adv_pred,alpha=0.8)
        #create the legend
        reshaped_seg=adv_pred.reshape(-1,adv_pred.shape[-1]).tolist()
        #unique values
        values=[list(x) for x in set(tuple(x) for x in reshaped_seg)]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=values[i],label=ref_dict.get(str(values[i])) ) for i in range(len(values))  ]
        # put those patched as legend-handles into the legend
        axarray[1,2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        
        if mIOUs==None:
            axarray[1,2].set_title("DLV3 Adversarial Prediction")
        else:
            axarray[1,2].set_title("mIOU = %.3f"%(mIOUs[1]))

        plt.tight_layout()
        plt.show(block=True)
        if save:
            pylab.savefig("./saved_plots/"+title+".png")











    def plot_entropy(self,entropy,save=True,title="entropy",c_name="YlGnBu"):
        plt.figure()
        plt.title(title)
        plt.imshow(entropy,cmap=c_name,vmax=entropy.max(),vmin=entropy.min()) # if we add the image in the background then change alpha to 0.8
        # plt.imshow(img)
        plt.axis("off")
        plt.colorbar()
        plt.show(block=True)
        pylab.savefig("./saved_plots/"+title+".png")


    def plot_entropy_2x2(self,entropy,adv_entropy,save=True,title="entropy",c_name="YlGnBu"):
        fig,axarray=plt.subplots(nrows=1,ncols=2)

        im1=axarray[0].imshow(entropy,cmap=c_name,vmax=entropy.max(),vmin=entropy.min())
        axarray[0].axis("off")
        axarray[0].set_title("Image")
        divider=make_axes_locatable(axarray[0])
        cax1=divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1,ax=axarray[0],cax=cax1)

        im2=axarray[1].imshow(adv_entropy,cmap=c_name,vmax=adv_entropy.max(),vmin=adv_entropy.min())
        axarray[1].axis("off")
        axarray[1].set_title("Perturbed Image")
        divider2=make_axes_locatable(axarray[1])
        cax2=divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2,ax=axarray[1],cax=cax2)

        plt.show(block=True)
        if save:
            pylab.savefig("./saved_plots/"+title+".png")


    def clear_saved_plots(self):
        #delete all images from saved_plots directory 
        delete_old_imgs=True
        if delete_old_imgs:
            filelist = [ f for f in os.listdir("./saved_plots") if f.endswith(".png") ]
            if filelist!=[]:
                print("--Removing old images from saved_plots")
                for f in filelist:
                    os.remove(os.path.join("./saved_plots", f))


    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv')

