import torch
import torchvision

from torchvision import transforms

import os
from PIL import Image

class FreiburgCarsDataset(torch.utils.data.Dataset):
    """Freiburg cars dataset."""

    def __init__(self,
                    path,
                    imsize=None, 
                    segment=True, 
                    crop=False,
                    n_bins=8,
                    **kwargs):

        
        self.src_dir = os.path.join(path, "images")
        self.lab_dir = os.path.join(path, "annotations")
        self.mask_dir = os.path.join(path, "instance_masks")

        self.ids = sorted(os.listdir(self.src_dir))
        self.images = sorted([obj_id+'/'+im_id for obj_id in self.ids for im_id in os.listdir(os.path.join(self.src_dir, obj_id))])

        self.len = len(self.images)
        self.imsize = imsize
        self.segment = segment
        self.crop = crop

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.resize = transforms.Resize(imsize, antialias=True)
        self.prep_im = transforms.Compose([
#            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.n_bins = n_bins

        self.geom_augment = \
            transforms.RandomResizedCrop(imsize, scale=(.5,1.5), interpolation=Image.BICUBIC, antialias=True)




    def instance_segment_images(self, CUDA=True):

        if not os.path.exists(self.mask_dir):
            os.mkdir(self.mask_dir)

        for image in self.images:
            img_name = os.path.join(self.src_dir, image)

            mask_name = img_name.replace(self.src_dir, self.mask_dir)
            if not os.path.exists(os.path.dirname(mask_name)):
                os.mkdir(os.path.dirname(mask_name))
            elif os.path.exists(mask_name):
                continue

            image = Image.open(img_name).convert('RGB')
            h, w = image.width, image.height
            image_tensor = torchvision.transforms.ToTensor()(image)
            model = self.instance_seg_model
            if CUDA:
                image_tensor = image_tensor.cuda()
                model.cuda()
            if h*w > 1e6:
                # Dirty hack in case the image might cause GPU OOM
                image_tensor = image_tensor.cpu()
                model.cpu()
            out = model([image_tensor])[0]
            boxes = out['boxes']
            largest_box = np.argmax([(x2-x1) * (y2-y1) for x1, y1, x2, y2 in boxes])
            mask = out['masks'][largest_box]
            torchvision.utils.save_image(mask.float(), mask_name)

        for model in self.segmentation_models:
            model.cpu()


    def get_annotations(self, im_id):
        car_id = im_id.split('/')[-2]
        car_num = str(int(car_id[-2:]))
        img_name = car_id + '/' + im_id.split('/')[-1]
        annotations_name = os.path.join(self.src_dir, car_num+'_annot.txt').replace(self.src_dir, self.lab_dir)
        with open(annotations_name) as f:
            for line in f:
                name, x1, y1, x2, y2, az = line.split()
                if name[:-4] == img_name[:-4]:
                    break
        bbox = int(x1), int(y1), int(x2), int(y2)
        return bbox, int(az)

    def az2vp_id(self, az):
        az = az%360
        return(int(az / 360 * self.n_bins))

    def __getids__(self):
        return self.ids


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        img_path = os.path.join(self.src_dir,
                                self.images[idx])
        pil_image = Image.open(img_path).convert('RGB')
        img = transforms.functional.to_tensor(pil_image)

        mask_path = img_path.replace(self.src_dir, self.mask_dir)
        mask = Image.open(mask_path).convert('L')
        mask = transforms.functional.to_tensor(mask)
        mask = (mask>.5).float()

        _, az = self.get_annotations(img_path)
        vp = self.az2vp_id(az)
        
        if self.crop:
            bbox = _get_bbox(mask[0])
            img = transforms.functional.crop(img, *bbox)
            mask = transforms.functional.crop(mask, *bbox)

        combined = torch.cat([img, mask], dim=0)
        aug_combined = self.geom_augment(combined)
        img, mask = aug_combined[:3], aug_combined[3:]

        img = self.resize(img)
        mask = self.resize(mask)

        img = self.prep_im(img)

        return {'img': img, 'mask': mask, 'idx': idx, 'vp': vp, 'cat': 0}

def _get_bbox(mask):
    horizontal_indicies = (torch.any(mask, dim=-2) != 0).nonzero()
    vertical_indicies = (torch.any(mask, dim=-1) != 0).nonzero()
    h0, h1 = horizontal_indicies[0], horizontal_indicies[-1]
    v0, v1 = vertical_indicies[0], vertical_indicies[-1]

    return v0, h0, v1-v0, h1-h0
