import os
import torch
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
from model import Model
import pytorch_iou


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


IOU = pytorch_iou.IOU(size_average=True)


def compute_loss(out, saliency, label):
    """
    Args:
        out:  [B, 1, H, W]
        saliency:  [B, 1, H, W]
        label:  [B, 1, H, W]
    """
   
    out = F.interpolate(out, label.shape[2:], mode='bilinear', align_corners=False)
    saliency = F.interpolate(saliency, label.shape[2:], mode='bilinear', align_corners=False)

 
    loss_out = F.binary_cross_entropy_with_logits(out, label, reduction='mean')


    loss_saliency = F.binary_cross_entropy_with_logits(saliency, label, reduction='mean')


    out_sig = torch.sigmoid(out)
    loss_iou = IOU(out_sig, label)


    total_loss = loss_out + 0.5 * loss_saliency + 0.5 * loss_iou

    return total_loss, loss_out, loss_saliency, loss_iou


if __name__ == '__main__':

    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    torch.cuda.manual_seed(2021)
    torch.cuda.manual_seed_all(2021)


    img_root = '' 
    save_path = ''  
    pretrained_path = '' 


    lr = 1e-4  
    batch_size = 4
    num_epochs = 75
    lr_decay_epochs = [60, 80]  
    save_epoch_start = 50  
    print_freq = 50 

  
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    print("Loading dataset...")
    data = Data(img_root, mode='train')
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    iter_num = len(loader)
    print(f"Dataset loaded: {len(data)} images, {iter_num} iterations per epoch")

  
    print("Initializing model...")
    net = Model().cuda()

   
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        net.load_pretrain_model(pretrained_path)


    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")



    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )


    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


    print("Starting training...")
    print("=" * 60)

    net.train()
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):

        if epoch in lr_decay_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"Learning rate decayed to {optimizer.param_groups[0]['lr']}")


        prefetcher = DataPrefetcher(loader)
        rgb, t, label = prefetcher.next()


        running_loss = 0.0
        running_loss_out = 0.0
        running_loss_sal = 0.0
        running_loss_iou = 0.0
        iteration = 0


        while rgb is not None:
            iteration += 1


            out, saliency = net(rgb, t)

            total_loss, loss_out, loss_sal, loss_iou = compute_loss(out, saliency, label)


            optimizer.zero_grad()
            total_loss.backward()

  
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()


            running_loss += total_loss.item()
            running_loss_out += loss_out.item()
            running_loss_sal += loss_sal.item()
            running_loss_iou += loss_iou.item()


            if iteration % print_freq == 0:
                avg_loss = running_loss / print_freq
                avg_loss_out = running_loss_out / print_freq
                avg_loss_sal = running_loss_sal / print_freq
                avg_loss_iou = running_loss_iou / print_freq
                current_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch: [{epoch:3d}/{num_epochs}] | Iter: [{iteration:4d}/{iter_num}] | "
                      f"Loss: {avg_loss:.4f} (Out: {avg_loss_out:.4f}, Sal: {avg_loss_sal:.4f}, IOU: {avg_loss_iou:.4f}) | "
                      f"LR: {current_lr:.6f}")


                running_loss = 0.0
                running_loss_out = 0.0
                running_loss_sal = 0.0
                running_loss_iou = 0.0


            rgb, t, label = prefetcher.next()


        scheduler.step()


        if epoch >= save_epoch_start:
            save_file = os.path.join(save_path, f'epoch_{epoch}.pth')
            torch.save(net.state_dict(), save_file)
            print(f"Model saved to {save_file}")


        epoch_avg_loss = running_loss / max(1, iteration % print_freq)
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            best_model_path = os.path.join(save_path, 'best_model.pth')
            torch.save(net.state_dict(), best_model_path)
            print(f"Best model updated at epoch {epoch} with loss {best_loss:.4f}")

        print("-" * 60)


    final_model_path = os.path.join(save_path, 'final_model.pth')
    torch.save(net.state_dict(), final_model_path)
    print("=" * 60)
    print(f"Training completed! Final model saved to {final_model_path}")
