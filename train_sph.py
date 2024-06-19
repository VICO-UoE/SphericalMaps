import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import random

import os
import argparse
import yaml
import time
import shutil
from tqdm import tqdm

from datasets.FCars import FreiburgCarsDataset
from datasets.SPair import SPairDataset

from dino_mapper import (
    DINOMapper,
    triplet_distance_loss, 
    triplet_orientation_loss, 
    relative_viewpoint_loss, 
    )


def train_epoch(epoch, data, model, optim, cfg):

    n_triplets = cfg['model']['n_triplets']
    margin = cfg['model']['rd_margin']
    thresh = cfg['model']['o_thresh']
    rd_strength = cfg['model']['rd_strength']
    o_strength = cfg['model']['o_strength']
    vp_strength = cfg['model']['vp_strength']

    model = model.train()
    rec_epoch = 0
    orientation_epoch = 0
    distance_epoch = 0
    viewpoint_epoch = 0

    for it, batch in enumerate(pbar_batch := tqdm(data, desc='batch', dynamic_ncols=True)):
        loss = 0
        im, gt_mask, vp, cats = batch['img'], batch['mask'], batch['vp'], batch['cat']
        im = im.to(device)
        gt_mask = gt_mask.to(device)
#        vp = vp.to(device)
        cats = cats.to(device)

        b_size, _ ,h, w = im.shape
        feature_map_shape = (h//model.patch_size, w//model.patch_size)

        gt_mask = nn.functional.interpolate(gt_mask, feature_map_shape, mode='nearest').permute(0,2,3,1).reshape(b_size, -1, 1)
        
        feats, sph, _ = model(im, cats=cats, gt_mask=gt_mask)

        vp_loss = relative_viewpoint_loss(sph, vp, n_bins=cfg['data']['vp_bins'])

        cond = F.one_hot(cats, num_classes=cfg['data']['n_cats']).float()
        
        sph_loss = model.sphere_loss_prototypes_implicit(feats, sph, cond,)

        o_loss = triplet_orientation_loss(sph, fm_shape=feature_map_shape, 
                                            correct_sine=False, n_triplets=n_triplets, thresh=thresh)

        rd_loss = triplet_distance_loss(sph, fm_shape=feature_map_shape, 
                                            correct_sine=False, n_triplets=n_triplets, margin=margin)

        loss += vp_strength * vp_loss
        loss += o_strength * o_loss
        loss += rd_strength * rd_loss
        loss += 1e0 * sph_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        rec_epoch = (rec_epoch * it + sph_loss.cpu().detach().item()) / (it + 1)
        orientation_epoch = (orientation_epoch * it + o_loss.cpu().detach().item()) / (it + 1)
        distance_epoch = (distance_epoch * it + rd_loss.cpu().detach().item()) / (it + 1)
        viewpoint_epoch = (viewpoint_epoch * it + vp_loss.cpu().detach().item()) / (it + 1)

        pbar_batch.set_description(f"e: {epoch},r: {rec_epoch:.3f}, o: {orientation_epoch:.3f}, d: {distance_epoch:.3f}, v: {viewpoint_epoch:.3f}")


    with open(exp_dir+'/logs/log.txt', 'a') as log:
        log.write(f"epoch: {epoch}, reconstruction_loss: {rec_epoch:.3f}, orientation_loss: {orientation_epoch:.3f}, distance_loss: {distance_epoch:.3f}, viewpoint_loss: {viewpoint_epoch:.3f}\n")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(cfg):

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    model = DINOMapper(backbone=cfg['model']['backbone'], n_cats=cfg['data']['n_cats'])
    ckpt_path = cfg['training']['resume_from_ckpt']
    if ckpt_path:
        print("Loading checkpoint from", ckpt_path)
        model.load_checkpoint(ckpt_path, device)

    optimizer = torch.optim.Adam([{'params': model.sphere_mapper.parameters()},
                                  {'params': model.prototypes.parameters()},],
                                 lr=cfg['training']['learning_rate']) 
                                   

    epochs = cfg['training']['epochs']
    batch_size = cfg['training']['batch_size']
    assert cfg['data']['set'] in ['SPair', 'FCars'], f"""{cfg['data']['set']} is not a valid training set"""
    if cfg['data']['set'] == 'SPair':
        TrainSet = SPairDataset
    elif cfg['data']['set'] == 'FCars':
        TrainSet = FreiburgCarsDataset
    train_dataset = TrainSet(path=cfg['data']['data_path'], resize_im=True, imsize=cfg['data']['im_size'], n_bins=cfg['data']['vp_bins'], training_batch=True, replications=10)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, drop_last=False, shuffle=True, worker_init_fn=seed_worker, generator=g,)

    model = model.to(device)
    e = 0

    for e in (pbar_epoch := tqdm(range(epochs), dynamic_ncols=True)):
        pbar_epoch.set_description("Epoch: %d"%e)

        train_epoch(epoch=e, data=train_dataloader, model=model, optim=optimizer, cfg=cfg)

        if (e+1)%cfg['logs']['save_frequency']==0:
            model.make_checkpoint(exp_dir+f'/ckpts/{e+1}.pth')

    print('Done')





def init_exps_dirs(path):
    start_time = time.gmtime()
    exp_dir = path+'exp_'+str(device)+'_{:0>2}_{:0>2}_{:0>2}_{:0>2}:{:0>2}:{:0>2}'.format(
                   str(start_time.tm_year),
                   str(start_time.tm_mon),
                   str(start_time.tm_mday),
                   str(start_time.tm_hour),
                   str(start_time.tm_min),
                   str(start_time.tm_sec))
    print('Creating log directory at '+exp_dir)
    os.makedirs(exp_dir)
    os.makedirs(exp_dir+'/logs')
    os.makedirs(exp_dir+'/plots/train')
    os.makedirs(exp_dir+'/ckpts')
    return exp_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to configuration file')

    args = parser.parse_args()
    with open(args.config) as config_file:
        cfg = yaml.safe_load(config_file)

    global seed
    seed = cfg['training']['seed']
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    global device
    dev = cfg['training']['device']
    if torch.cuda.is_available() and isinstance(dev, int) and dev>=0:
        print('Using GPU '+str(dev))
        device = torch.device('cuda', dev)
    else:
        print('Using CPU')
        device = torch.device('cpu')

    global exp_dir
    if cfg['logs']['enabled']:
        exp_dir = init_exps_dirs(cfg['logs']['base_dir'])
        shutil.copy(args.config, exp_dir+'/conf.yml')

    train(cfg)
