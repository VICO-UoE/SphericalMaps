import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import numpy as np
import shutil

import json
import os
from pathlib import Path
from PIL import Image

class SPairDataset(torch.utils.data.Dataset):
    def __init__(self, 
                    path,
                    split='trn', 
                    category=None, 
                    resize_im=False, 
                    imsize=(224,224),
                    bbox_crop=False, 
                    training_batch=False, 
                    replications=1,
                    use_resegmanted_images=True,
                    **kwargs):
        self.path = path
        self.split = split
        self.pairs_path = os.path.join(path, "PairAnnotation", split)
        self.imgs_path = os.path.join(path, "JPEGImages")
        self.pairs = sorted(os.listdir(self.pairs_path))
        if category is not None:
            self.pairs = [pair for pair in self.pairs if category in pair]

        src_ims = set([p.split('-')[1] for p in self.pairs] + [p.split('-')[2].split(':')[0] for p in self.pairs])
        self.src_ids = {idx: i for i,idx in enumerate(src_ims)}


        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.resize_im = resize_im
        self.resize = transforms.Resize(imsize, antialias=True)
        self.bbox_crop = bbox_crop
        self.training_batch = training_batch
        if training_batch:
            registered_obj = []
            registered_pairs = []
            for pair in self.pairs:
                obj_id = pair.split('/')[-1].split('-')[1]
                if obj_id in registered_obj:
                    continue
                else:
                    registered_obj.append(obj_id)
                    registered_pairs.append(pair)
            self.pairs = registered_pairs
        self.pairs = self.pairs * replications


        self.cat_dict = {
        'aeroplane':    0,
        'bicycle':      1,
        'bird':         2,
        'boat':         3,
        'bottle':       4,
        'bus':          5,
        'car':          6,
        'cat':          7,
        'chair':        8,
        'cow':          9,
        'dog':         10,
        'horse':       11,
        'motorbike':   12,
        'person':      13,
        'pottedplant': 14,
        'sheep':       15,
        'train':       16,
        'tvmonitor':   17,
        }
        self.n_cats = len(self.cat_dict)

        self.urs = use_resegmanted_images
        if self.urs:
            self._preprocess_segmentation()

            
        self.geom_augment = transforms.RandomResizedCrop(imsize, scale=(.5,1.5), interpolation=Image.BICUBIC, antialias=True)
        self.color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):

        ann_path = os.path.join(self.pairs_path, self.pairs[idx])
        with open(ann_path, 'r') as ann_file:
            annotations = json.load(ann_file)

        cat = annotations['category']
        src_img = annotations['src_imname']
        src_bbx = annotations['src_bndbox']
        src_kps = annotations['src_kps']

        trg_img = annotations['trg_imname']
        trg_bbx = annotations['trg_bndbox']
        trg_kps = annotations['trg_kps']

        src_ann_path = os.path.join(self.path, 'ImageAnnotation', cat, src_img.replace('.jpg', '.json'))
        with open(src_ann_path, 'r') as src_ann_file:
            src_annotations = json.load(src_ann_file)

        src_vp = src_annotations['azimuth_id']
#        src_ang = torch.tensor(src_vp * np.pi/4, dtype=torch.float32).item()
        src_all_kps = src_annotations['kps']

        trg_ann_path = os.path.join(self.path, 'ImageAnnotation', cat, trg_img.replace('.jpg', '.json'))
        with open(trg_ann_path, 'r') as trg_ann_file:
            trg_annotations = json.load(trg_ann_file)

        trg_vp = trg_annotations['azimuth_id']
        trg_all_kps = trg_annotations['kps']

        #Remove None from kps dicts
        src_all_kps = torch.tensor([v if v is not None else [-1,-1] for v in src_all_kps.values()]).flip(-1)
        trg_all_kps = torch.tensor([v if v is not None else [-1,-1] for v in trg_all_kps.values()]).flip(-1)


        src_im_path = os.path.join(self.imgs_path, cat, src_img)
        src_im = Image.open(src_im_path).convert('RGB')
        src_shape = src_im.size

        trg_im_path = os.path.join(self.imgs_path, cat, trg_img)
        trg_im = Image.open(trg_im_path).convert('RGB')
        trg_shape = trg_im.size

        src_kps = torch.from_numpy(np.array(src_kps)).flip(-1)
        trg_kps = torch.from_numpy(np.array(trg_kps)).flip(-1)

        alpha = max(trg_bbx[2] - trg_bbx[0], trg_bbx[3] - trg_bbx[1])

        seg_dir = 'Re_segmentation' if self.urs else 'Segmentation'
        src_mask_path = src_im_path.replace('JPEGImages', seg_dir).replace('.jpg', '.png')
        src_mask = Image.open(src_mask_path).convert('L')
        src_mask = transforms.functional.to_tensor(src_mask)
        src_mask = (src_mask > .5).float()
        trg_mask_path = trg_im_path.replace('JPEGImages', seg_dir).replace('.jpg', '.png')
        trg_mask = Image.open(trg_mask_path).convert('L')
        trg_mask = transforms.functional.to_tensor(trg_mask)
        trg_mask = (trg_mask > .5).float()

        src_id = self.src_ids[src_img.split('.')[0]]
        trg_id = self.src_ids[trg_img.split('.')[0]]

        if self.training_batch:
            if self.bbox_crop:
                left = src_bbx[0]
                top = src_bbx[1]
                width = src_bbx[2] - src_bbx[0]
                height = src_bbx[3] - src_bbx[1]
                src_im = transforms.functional.crop(src_im, top, left, height, width)
                src_mask = transforms.functional.crop(src_mask, top, left, height, width)
            if self.resize_im:
#                src_im = self.color_jittering(src_im)
                src_im = self.to_tensor(src_im)
                combined = torch.cat([src_im, src_mask], dim=0)
                aug_combined = self.geom_augment(combined)
                src_im, src_mask = aug_combined[:3], aug_combined[3:]
                src_im = self.normalize(src_im)
            return {'img': src_im, 'mask': src_mask, 'idx': src_id, 'vp': src_vp, 'cat': self.cat_dict[cat]}

        src_im = self.resize(src_im)
        trg_im = self.resize(trg_im)
        src_im = self.to_tensor(src_im)
        trg_im = self.to_tensor(trg_im)
        src_im = self.normalize(src_im)
        trg_im = self.normalize(trg_im)

        return {'src_im': src_im,
                'trg_im': trg_im,
                'src_kps': src_kps, 
                'trg_kps': trg_kps, 
                'src_all_kps': src_all_kps, 
                'trg_all_kps': trg_all_kps, 
                'src_mask': src_mask, 
                'trg_mask': trg_mask, 
                'src_shape': np.array(src_shape),
                'trg_shape': np.array(trg_shape),
                'bbox': np.array(trg_bbx), 
                'src_id': src_id, 
                'trg_id': trg_id,
                'cat': self.cat_dict[cat]}

    def _preprocess_segmentation(self, CUDA=True, device='cuda'):

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        cat_dict = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
        cat_dict['aeroplane'] = cat_dict['airplane']
        cat_dict['motorbike'] = cat_dict['motorcycle']
        cat_dict['pottedplant'] = cat_dict['potted plant']
        cat_dict['tvmonitor'] = cat_dict['tv']
        model = maskrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        model.eval()

        reseg_mask_dir = self.imgs_path.replace('JPEGImages', 'Re_segmentation')

        if not os.path.exists(reseg_mask_dir):
            os.mkdir(reseg_mask_dir)

        ims = set()

        for pair in self.pairs:
            ann_path = os.path.join(self.pairs_path, pair)
            with open(ann_path, 'r') as ann_file:
                annotations = json.load(ann_file)

            cat = annotations['category']
            src_img = annotations['src_imname']
            trg_img = annotations['trg_imname']

            ims.add((cat,src_img))
            ims.add((cat,trg_img))

        for cat, img in ims:

            img_path = os.path.join(self.imgs_path, cat, img)
            new_mask_path = img_path.replace(self.imgs_path, reseg_mask_dir).replace('.jpg', '.png')

            if not os.path.exists(os.path.dirname(new_mask_path)):
                os.mkdir(os.path.dirname(new_mask_path))
            elif os.path.exists(new_mask_path):
                continue

            image = Image.open(img_path).convert('RGB')
            h, w = image.width, image.height
            image_tensor = torchvision.transforms.ToTensor()(image)
            if CUDA:
                image_tensor = image_tensor.to(device)
                model.to(device)
            if h*w > 1e6:
                # Dirty hack in case the image might cause GPU OOM
                image_tensor = image_tensor.cpu()
                model.cpu()

            ###
            out = model([image_tensor])[0]
            boxes = out['boxes']
            labels = out['labels']
            masks = out['masks']
            matches = labels == cat_dict[cat]
            if not matches.any(): #detection failed, use provided mask
                old_mask_path = img_path.replace('JPEGImages', 'Segmentation').replace('.jpg', '.png')
                mask = Image.open(old_mask_path).convert('L')
                mask = transforms.functional.to_tensor(mask)
                mask = (mask > 0).float()
            else:
                first_match = matches.nonzero()[0]
                mask = masks[first_match]
            torchvision.utils.save_image(mask.float(), new_mask_path)

        del model
