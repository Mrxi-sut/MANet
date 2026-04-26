import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from model import Model
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    model_path = '' 
    out_path = ''  
    data_root = '' 

    save_saliency = True 
    batch_size = 1  

   
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f'Created output directory: {out_path}')

    if save_saliency:
        saliency_path = os.path.join(out_path, 'saliency')
        if not os.path.exists(saliency_path):
            os.makedirs(saliency_path)
            print(f'Created saliency directory: {saliency_path}')

  
    print('Loading test dataset...')
    data = Data(root=data_root, mode='test')
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f'Total test images: {len(data)}')

   
    print('Building model...')
    net = Model().cuda()

  
    num_params = sum(p.numel() for p in net.parameters())
    print(f'Model parameters: {num_params:,} ({num_params/1e6:.2f}M)')

    if not os.path.exists(model_path):
        print(f'Error: Model file not found at {model_path}')
        exit(1)

    print(f'Loading model weights from: {model_path}')
    net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    print('Model loaded successfully!')


    print('=' * 60)
    print('Starting inference...')
    print('=' * 60)

    net.eval()
    time_start = time.time()
    img_num = len(loader)

    with torch.no_grad():
        for i, (rgb, t, _, (H, W), name) in enumerate(loader):
            
            print(f'[{i+1}/{img_num}] Processing: {name[0]}')

      
            rgb = rgb.cuda().float()
            t = t.cuda().float()

       
            out, saliency = net(rgb, t)

         
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        
            pred = torch.sigmoid(out)
            pred = pred.squeeze().cpu().data.numpy()  # [H, W]

         
            save_name = name[0]
            if not (save_name.endswith('.png') or save_name.endswith('.jpg')):
                save_name = save_name + '.png'
            else:
                save_name = save_name[:-4] + '.png'

   
            pred_uint8 = (pred * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_path, save_name), pred_uint8)

            if save_saliency:
                saliency_resized = F.interpolate(saliency, size=(H, W), mode='bilinear', align_corners=False)
                saliency_map = torch.sigmoid(saliency_resized)
                saliency_map = saliency_map.squeeze().cpu().data.numpy()
                saliency_uint8 = (saliency_map * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(saliency_path, save_name), saliency_uint8)

  
    time_end = time.time()
    total_time = time_end - time_start

    print('=' * 60)
    print('Testing completed!')
    print(f'Total images processed: {img_num}')
    print(f'Total time: {total_time:.2f} seconds')
    print(f'Results saved to: {out_path}')
    if save_saliency:
        print(f'Saliency maps saved to: {saliency_path}')
    print('=' * 60)
