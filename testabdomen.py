import ssl
import os
import numpy as np
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
import nibabel as nib
from torch.autograd import Variable
from math import exp
import logging
import warnings

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="torchvision.io.image")
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
from utils import getters, setters
from utils.mappers import label2text_dict_abdomenct as label2text_dict
from utils.functions import (AverageMeter, registerSTModel, dice_eval, dice_binary, compute_HD95, jacobian_determinant)
from utils.loss import NccLoss


def extract_unigradicon_flow(model, moving, fixed, debug=False):
    """
    Extract initial displacement field from UniGradICON model (in voxels)

    Args:
        model:  Pre-trained UniGradICON model instance
        moving: Moving image [B, C, H, W, D]
        fixed:  Fixed image [B, C, H, W, D]

    Returns:
        flow:   Displacement field [B, 3, H, W, D] (in voxels)
    """
    original_shape = moving.shape[2:]
    expected_shape = (175, 175, 175)

    # Adjust input size to the model's expected size
    if original_shape != expected_shape:
        if debug:
            print(f"Adjusting input size: {original_shape} → {expected_shape}")
        moving_in = F.interpolate(moving, size=expected_shape, mode='trilinear', align_corners=False)
        fixed_in = F.interpolate(fixed, size=expected_shape, mode='trilinear', align_corners=False)
    else:
        moving_in = moving
        fixed_in = fixed

    with torch.no_grad():
        _ = model(moving_in, fixed_in)
        phi_AB = model.phi_AB_vectorfield  # Deformation field in normalized coordinate system
        identity = model.identity_map

        flow_normalized = phi_AB - identity

        # Convert to voxel units
        D, H, W = phi_AB.shape[2:]
        scaling = torch.tensor(
            [D - 1, H - 1, W - 1],
            device=flow_normalized.device,
            dtype=flow_normalized.dtype
        ).view(1, 3, 1, 1, 1)
        flow_voxel = flow_normalized * scaling

    # Scale the flow field back to the original image size
    if original_shape != expected_shape:
        if debug:
            print(f"Adjusting flow field back to original size: {expected_shape} → {original_shape}")
        flow_voxel = F.interpolate(flow_voxel, size=original_shape, mode='trilinear', align_corners=False)
        for i in range(3):
            flow_voxel[:, i] *= original_shape[i] / expected_shape[i]

    if debug:
        print(f"  [UniGradICON] Flow shape: {flow_voxel.shape}")
        print(f"  [UniGradICON] Flow range: [{flow_voxel.min():.3f}, {flow_voxel.max():.3f}]")

    return flow_voxel


class Grad3d(nn.Module):
    """First-order gradient regularization loss"""

    def __init__(self, penalty='l1', loss_mult=None):
        super().__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy, dx, dz = dy * dy, dx * dx, dz * dz

        grad = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0
        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window_3D(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t())
    _3D = _1D.mm(_2D.reshape(1, -1)).reshape(
        window_size, window_size, window_size
    ).float().unsqueeze(0).unsqueeze(0)
    return Variable(_3D.expand(channel, 1, window_size, window_size, window_size).contiguous())


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class SSIM3D(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window, self.channel = window, channel
        return 1 - _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


class MultiScaleNCC(nn.Module):
    """Multi-scale normalized cross-correlation loss"""

    def __init__(self, window_sizes=(5, 9, 13), weights=None):
        super().__init__()
        self.ncc_losses = [NccLoss([ws, ws, ws]) for ws in window_sizes]
        weights = weights or [0.5, 0.3, 0.2]
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def forward(self, fixed, moving):
        return sum(w * loss(fixed, moving) for loss, w in zip(self.ncc_losses, self.weights))


def unsupervised_optimization(moving, fixed, initial_flow, current_shape, config, debug=False):
    """
    Unsupervised test-time instance optimization

    Args:
        moving:        Moving image [B, C, H, W, D]
        fixed:         Fixed image [B, C, H, W, D]
        initial_flow:  Initial displacement field [B, 3, H, W, D]
        current_shape: Image spatial dimensions (H, W, D)
        config:        Optimization configuration dictionary

    Returns:
        disp_hr: Optimized displacement field [B, 3, H, W, D]
    """
    device = moving.device
    H, W, D = current_shape

    max_iterations = config.get('max_iterations', 10)
    initial_lr = config.get('learning_rate', 0.01)
    grid_sp_adam = config.get('grid_sp_adam', 1)
    lambda_sim = config.get('lambda_sim', 1.0)
    lambda_reg = config.get('lambda_reg', 1.0)
    lambda_ssim = config.get('lambda_ssim', 2.0)

    with torch.enable_grad():
        # Initialize optimizable displacement field (stored as Conv3d weights)
        initial_flow_lr = F.interpolate(
            initial_flow.detach(),
            size=(H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
            mode='trilinear', align_corners=False
        )
        net = nn.Sequential(
            nn.Conv3d(3, 1, (H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam), bias=False)
        )
        net[0].weight.data[:] = (initial_flow_lr / grid_sp_adam).float().cpu().data
        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=False
        )

        with torch.no_grad():
            grid0 = F.affine_grid(
                torch.eye(3, 4).unsqueeze(0).to(device),
                (1, 1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam),
                align_corners=False
            )
            scale = torch.tensor([
                (H // grid_sp_adam - 1) / 2,
                (W // grid_sp_adam - 1) / 2,
                (D // grid_sp_adam - 1) / 2
            ]).to(device)
            patch_image_fix = F.avg_pool3d(fixed, grid_sp_adam, stride=grid_sp_adam)

        ncc_loss_fn = MultiScaleNCC(window_sizes=(5, 9, 13))
        reg_loss_fn = Grad3d()
        ssim_loss_fn = SSIM3D()

        best_loss = float('inf')
        best_weight = net[0].weight.data.clone()
        patience_counter = 0
        max_patience = 3

        for it in range(max_iterations):
            optimizer.zero_grad()

            disp_sample = net[0].weight.permute(0, 2, 3, 4, 1)
            grid_disp = (grid0.view(-1, 3).to(device).float()
                         + (disp_sample.view(-1, 3) / scale).flip(1).float())
            grid_reshaped = grid_disp.view(
                1, H // grid_sp_adam, W // grid_sp_adam, D // grid_sp_adam, 3
            ).to(device)

            warped_image = F.grid_sample(
                F.avg_pool3d(moving, grid_sp_adam, stride=grid_sp_adam).float(),
                grid_reshaped, align_corners=False, mode='bilinear'
            )

            ncc_loss = lambda_sim * ncc_loss_fn(patch_image_fix, warped_image)
            reg_loss = lambda_reg * reg_loss_fn(net[0].weight)
            ssim_loss = lambda_ssim * ssim_loss_fn(patch_image_fix, warped_image)
            total_loss = ncc_loss + reg_loss + ssim_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_weight = net[0].weight.data.clone()
                patience_counter = 0
                if debug and it % 2 == 0:
                    print(f"     iter {it:2d}: loss={current_loss:.4f} "
                          f"(ncc={ncc_loss.item():.3f}, reg={reg_loss.item():.3f}, "
                          f"ssim={ssim_loss.item():.3f}) ★")
            else:
                patience_counter += 1
                if debug and it % 2 == 0:
                    print(f"     iter {it:2d}: loss={current_loss:.4f} "
                          f"(patience {patience_counter}/{max_patience})")

            if patience_counter >= max_patience:
                if debug:
                    print(f"Early stopping: loss has not improved for {max_patience} consecutive iterations")
                break

            if it % 2 == 0:
                scheduler.step(current_loss)

        net[0].weight.data = best_weight
        with torch.no_grad():
            disp_hr = F.interpolate(
                net[0].weight * grid_sp_adam,
                size=(H, W, D), mode='trilinear', align_corners=False
            )
            if debug:
                print(f"  ✓ Optimization finished: final loss={best_loss:.4f}")

    return disp_hr


def unified_registration(moving, fixed, pretrained_model, config):
    """
    UniGradICON pre-trained initialization + test-time instance optimization registration

    Args:
        moving:           Moving image [B, C, H, W, D]
        fixed:            Fixed image [B, C, H, W, D]
        pretrained_model: Pre-trained UniGradICON model
        config:           Registration configuration dictionary

    Returns:
        initial_flow:      Initial displacement field [B, 3, H, W, D]
        optimized_flow:    Optimized displacement field [B, 3, H, W, D]
        initial_flow_time: Initial flow prediction time (seconds)
        optimization_time: Instance optimization time (seconds)
    """
    debug = config.get('debug', False)

    moving = moving.float().contiguous()
    fixed = fixed.float().contiguous()
    original_shape = moving.shape[2:]
    batch_size = moving.shape[0]

    if debug:
        print("\n" + "=" * 80)
        print("  Starting UniGradICON registration process")
        print(f"  Original size: {original_shape} | Model expected size: (175, 175, 175)")
        print("=" * 80)

    # Step 1: Initial flow prediction
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    initial_flow = extract_unigradicon_flow(pretrained_model, moving, fixed, debug)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    initial_flow_time = time.time() - t0

    # Step 2: Instance optimization
    if config.get('max_iterations', 0) > 0:
        if debug:
            print(f"  Starting instance optimization (max_iterations={config['max_iterations']})")
            print("-" * 80)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        optimized_flow = unsupervised_optimization(
            moving, fixed, initial_flow, original_shape, config, debug
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        optimization_time = time.time() - t1

        if debug:
            print(f"Instance optimization finished, time elapsed: {optimization_time:.4f}s")
    else:
        if debug:
            print("Skipping instance optimization (max_iterations=0)")
        optimized_flow = initial_flow
        optimization_time = 0.0

    # Unify dimensions
    if initial_flow.dim() == 4:
        initial_flow = initial_flow.unsqueeze(0)
    if optimized_flow.dim() == 4:
        optimized_flow = optimized_flow.unsqueeze(0)
    if initial_flow.shape[0] != batch_size:
        initial_flow = initial_flow.repeat(batch_size, 1, 1, 1, 1)
    if optimized_flow.shape[0] != batch_size:
        optimized_flow = optimized_flow.repeat(batch_size, 1, 1, 1, 1)

    if debug:
        print(f"  ✓ Registration complete | Initial flow: {initial_flow_time:.4f}s | Optimization: {optimization_time:.4f}s")

    return (
        initial_flow.contiguous(),
        optimized_flow.contiguous(),
        initial_flow_time,
        optimization_time,
    )


def run(opt):
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    test_loader = getters.getDataLoader(opt, split=opt['field_split'])

    print(f"Loading UniGradICON model: {opt['model']} ...")
    pretrained_model, _ = getters.getTestModelWithCheckpoints(opt)
    if torch.cuda.is_available():
        pretrained_model = pretrained_model.cuda().float()
    pretrained_model.eval()

    registration_config = {
        'debug': opt.get('debug', False),
        'learning_rate': opt.get('learning_rate', 0.01),
        'max_iterations': opt.get('max_iterations', 10),
        'grid_sp_adam': opt.get('grid_sp_adam', 1),
        'lambda_sim': opt.get('lambda_sim', 1.0),
        'lambda_reg': opt.get('lambda_reg', 1.0),
        'lambda_ssim': opt.get('lambda_ssim', 2.0),
    }

    print("\n" + "=" * 80)
    print(f"✓ Model: UniGradICON")
    print("Configuration parameters:")
    for k, v in registration_config.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    reg_model_ne = registerSTModel(opt['img_size'], 'nearest').cuda()
    reg_model_ti = registerSTModel(opt['img_size'], 'bilinear').cuda()

    organ_eval_dsc = [AverageMeter() for _ in range(1, 14)]
    eval_dsc = AverageMeter()
    init_dsc = AverageMeter()
    eval_det = AverageMeter()
    eval_std_det = AverageMeter()
    eval_hd95 = AverageMeter()
    init_hd95 = AverageMeter()

    total_initial_flow_time = 0.0
    total_optimization_time = 0.0
    total_registration_time = 0.0
    total_registrations = 0

    df_data = []

    with torch.no_grad():
        for num, data in enumerate(test_loader):
            idx1, idx2 = data[4][0].item(), data[5][0].item()
            data = [Variable(t.cuda()) for t in data[:10]]
            x, x_seg = data[0].float(), data[1].long()
            y, y_seg = data[2].float(), data[3].long()

            print("\n" + "=" * 80)
            print(f"Processing Pair ({idx1}, {idx2})")
            print("=" * 80)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            reg_start = time.time()

            initial_flow, pos_flow, init_time, opt_time = unified_registration(
                x, y, pretrained_model, registration_config
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            registration_time = time.time() - reg_start

            total_initial_flow_time += init_time
            total_optimization_time += opt_time
            total_registration_time += registration_time
            total_registrations += 1

            def_y = reg_model_ti(x, pos_flow)
            def_out = reg_model_ne(x_seg.float(), pos_flow)

            loop_df_data = [idx1, idx2]
            for idx in range(1, 14):
                dsc_idx = dice_binary(
                    def_out.long().squeeze().cpu().numpy(),
                    y_seg.long().squeeze().cpu().numpy(), idx
                )
                loop_df_data.append(dsc_idx)
                organ_eval_dsc[idx - 1].update(dsc_idx, x.size(0))

            dsc1 = dice_eval(def_out.long(), y_seg.long(), 14)
            eval_dsc.update(dsc1.item(), x.size(0))

            dsc2 = dice_eval(x_seg.long(), y_seg.long(), 14)
            init_dsc.update(dsc2.item(), x.size(0))

            jac_det = jacobian_determinant(pos_flow.detach().cpu().numpy())
            jac_det_val = np.sum(jac_det <= 0) / np.prod(x_seg.shape)
            eval_det.update(jac_det_val, x.size(0))

            log_jac_det = np.log(np.abs((jac_det + 3).clip(1e-8, 1e8)))
            std_dev_jac = np.std(log_jac_det)
            eval_std_det.update(std_dev_jac, x.size(0))

            moving_np = x_seg.long().squeeze().cpu().numpy()
            fixed_np = y_seg.long().squeeze().cpu().numpy()
            moving_warped = def_out.long().squeeze().cpu().numpy()

            hd95_1 = compute_HD95(moving_np, fixed_np, moving_warped, 14, np.ones(3) * 4)
            eval_hd95.update(hd95_1, x.size(0))

            hd95_2 = compute_HD95(moving_np, fixed_np, moving_np, 14, np.ones(3) * 4)
            init_hd95.update(hd95_2, x.size(0))

            improvement = dsc1.item() - dsc2.item()
            status = "Excellent" if dsc1.item() >= 0.75 else "Good" if dsc1.item() >= 0.70 else "Needs Tuning"
            print(f"Pair ({idx1}, {idx2}) | Registration effect: {status} | "
                  f"dice: {dsc1.item():.4f} ({improvement:+.4f}), init: {dsc2.item():.4f}, "
                  f"jac: {jac_det_val:.4f}, hd95: {hd95_1:.4f}, "
                  f"init_time: {init_time:.4f}s, opt_time: {opt_time:.4f}s, "
                  f"total_time: {registration_time:.4f}s")

            loop_df_data += [
                dsc1.item(), dsc2.item(), jac_det_val, std_dev_jac,
                hd95_1, hd95_2, init_time, opt_time, registration_time
            ]
            df_data.append(loop_df_data)

            # Save registration results
            if opt['is_save'] and num == 34:
                print(f"   Saving registration results (Pair {idx1}-{idx2})...")
                fp_flow = os.path.join('logs', opt['dataset'], 'UniGradICON_Optimized', 'flow_fields')
                fp_initial = os.path.join('logs', opt['dataset'], 'UniGradICON_Optimized', 'initial_flows')
                fp_warped = os.path.join('logs', opt['dataset'], 'UniGradICON_Optimized', 'warped_images')
                for p in [fp_flow, fp_initial, fp_warped]:
                    os.makedirs(p, exist_ok=True)

                prefix = f"{str(idx1).zfill(4)}_{str(idx2).zfill(4)}"

                nib.save(nib.Nifti1Image(initial_flow.permute(2, 3, 4, 1, 0).cpu().numpy(), None),
                         os.path.join(fp_initial, f'{prefix}_initial_flow.nii.gz'))
                nib.save(nib.Nifti1Image(pos_flow.permute(2, 3, 4, 1, 0).cpu().numpy(), None),
                         os.path.join(fp_flow, f'{prefix}_flow.nii.gz'))

                def_y_init = reg_model_ti(x, initial_flow)
                def_out_init = reg_model_ne(x_seg.float(), initial_flow)

                nib.save(nib.Nifti1Image(def_y_init[0, 0].cpu().numpy(), None),
                         os.path.join(fp_warped, f'{prefix}_warped_img_initial.nii.gz'))
                nib.save(nib.Nifti1Image(def_out_init[0, 0].long().float().cpu().numpy(), None),
                         os.path.join(fp_warped, f'{prefix}_seg_warped_initial.nii.gz'))
                nib.save(nib.Nifti1Image(def_y[0, 0].cpu().numpy(), None),
                         os.path.join(fp_warped, f'{prefix}_warped_img.nii.gz'))
                nib.save(nib.Nifti1Image(def_out[0, 0].long().float().cpu().numpy(), None),
                         os.path.join(fp_warped, f'{prefix}_seg_warped.nii.gz'))

                for name, tensor in [
                    ('img_moving', x[0, 0]),
                    ('img_fixed', y[0, 0]),
                    ('seg_moving', x_seg[0, 0].float()),
                    ('seg_fixed', y_seg[0, 0].float()),
                ]:
                    nib.save(nib.Nifti1Image(tensor.cpu().numpy(), None),
                             os.path.join(fp_warped, f'{prefix}_{name}.nii.gz'))

                print(f"  ✓ Registration results saved (Pair {idx1}-{idx2})")

    # ── Summary Statistics ──────────────────────────────────────────────────────────────
    overall_improvement = eval_dsc.avg - init_dsc.avg
    status = "Excellent" if eval_dsc.avg >= 0.75 else ("Good" if eval_dsc.avg >= 0.70 else "Needs Tuning")

    avg_init_time = total_initial_flow_time / total_registrations if total_registrations > 0 else 0
    avg_opt_time = total_optimization_time / total_registrations if total_registrations > 0 else 0
    avg_total_time = total_registration_time / total_registrations if total_registrations > 0 else 0

    print("\n" + "=" * 80)
    print("Final result statistics [UniGradICON]")
    print("=" * 80)
    print(f"{status}: Dice {init_dsc.avg:.4f} ➜ {eval_dsc.avg:.4f} ({overall_improvement:+.4f})")
    print("\nOrgan-wise Dice:")
    for idx in range(1, 14):
        print(f"  {label2text_dict.get(idx, f'Organ{idx}')}: {organ_eval_dsc[idx - 1].avg:.4f}")
    print(f"\nDeformation quality: jac_det={eval_det.avg:.6f}, std_dev={eval_std_det.avg:.4f}")
    print(f"Boundary precision: hd95={eval_hd95.avg:.4f}")
    print("\n" + "=" * 80)
    print("Time statistics")
    print("=" * 80)
    print(f"Total initial flow prediction time: {total_initial_flow_time:.3f}s")
    print(f"Total instance optimization time:   {total_optimization_time:.3f}s")
    print(f"Total full registration time:       {total_registration_time:.3f}s")
    print(f"Average initial flow time:          {avg_init_time:.4f}s/registration")
    print(f"Average instance optimization time: {avg_opt_time:.4f}s/registration")
    print(f"Average full registration time:     {avg_total_time:.4f}s/registration")
    print(f"Total number of registrations:      {total_registrations}")
    print("=" * 80)

    avg_organ_dsc = [organ_eval_dsc[i].avg for i in range(13)]
    df_data.append(
        [0, 0] + avg_organ_dsc + [
            eval_dsc.avg, init_dsc.avg, eval_det.avg, eval_std_det.avg,
            eval_hd95.avg, init_hd95.avg,
            avg_init_time, avg_opt_time, avg_total_time
        ]
    )

    keys = ['idx1', 'idx2'] + [label2text_dict[i] for i in range(1, 14)] + [
        'val_dice', 'init_dice', 'jac_det', 'std_dev', 'hd95', 'init_hd95',
        'init_time', 'opt_time', 'total_time'
    ]
    df = pd.DataFrame(df_data, columns=keys)
    fp = os.path.join('logs', opt['dataset'], 'results_UniGradICON_Optimized.csv')
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    df.to_csv(fp, index=False)
    print(f"\n✓ Results saved to: {fp}")


if __name__ == '__main__':
    opt = {
        'img_size': (96, 80, 128),
        'in_shape': (96, 80, 128),
        'logs_path': './logs',
        'num_workers': 4,
        'save_freq': 5,
        'n_checkpoints': 2,
    }

    parser = argparse.ArgumentParser(description="UniGradICON + Instance Optimization Framework (Abdomen CT Dataset)")
    parser.add_argument("-m", "--model", type=str, default='UniGradICON')
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-d", "--dataset", type=str, default='abdomenreg')
    parser.add_argument("--gpu_id", type=str, default='3')
    parser.add_argument("-dp", "--datasets_path", type=str, default="/data2/cl/datasets/")
    parser.add_argument("--field_split", type=str, default='test')
    parser.add_argument("--is_save", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--log", type=str, default="./logs/abdomenreg/UniGradICON")
    # Instance optimization parameters
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--lambda_sim", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_ssim", type=float, default=2.0)
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--grid_sp_adam", type=int, default=1)

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]: s.split('=')[1] for s in unknowns}

    print("\n" + "=" * 80)
    print("✓ UniGradICON + Instance Optimization Framework (Abdomen CT Dataset)")
    print("=" * 80)

    run(opt)