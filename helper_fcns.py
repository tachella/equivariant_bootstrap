import deepinv as dinv
import torch
import numpy as np
from torchvision.transforms.functional import rotate
from tqdm import tqdm
import torch.nn as nn
from huggingface_hub import hf_hub_download
import os

def mse(a, b):
    return (a - b).pow(2).reshape(a.shape[0], -1).mean(dim=1).detach().cpu().numpy()


class DnCNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        depth=20,
        bias=True,
        nf=64,
        device="cpu",
    ):
        super(DnCNN, self).__init__()

        self.depth = depth

        self.in_conv = nn.Conv2d(
            in_channels, nf, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='circular'
        )
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='circular')
                for _ in range(self.depth - 2)
            ]
        )
        self.out_conv = nn.Conv2d(
            nf, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode='circular'
        )

        self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

        if device is not None:
            self.to(device)

    def forward(self, x, sigma=None):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image
        :param float sigma: noise level (not used)
        """
        x1 = self.in_conv(x)
        x1 = self.nl_list[0](x1)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x1)
            x1 = self.nl_list[i + 1](x_l)

        return self.out_conv(x1) + x


def bootstrap(x, y, physics, model, MC, max_angle, max_shift, flip=True, verbose=False, compute_variance=False,
              debias=False, exact_rotations=False):
    xhat = model(y, physics)
    bstrap_mse = np.zeros((MC, x.shape[0]))

    xvar = torch.zeros_like(xhat)
    xmean = torch.zeros_like(xhat)

    for i in tqdm(range(MC), disable=not verbose):
        flag1 = False
        flag2 = False
        xg = xhat

        if max_shift > 0:
            di = int(np.random.randint(-max_shift, max_shift))
            dj = int(np.random.randint(-max_shift, max_shift))
            xg = torch.roll(xhat, shifts=(di, dj), dims=(2, 3))
        if max_angle > 0:
            if exact_rotations:
                for i in range(j):
                    xg = torch.rot90(xg, k=1, dims=(2, 3))
            else:
                xg = rotate(xg, np.random.randn() * max_angle)

        if flip:
            if np.random.rand() > .5:
                flag1 = True
                xg = torch.fliplr(xg)
            if np.random.rand() > .5:
                xg = torch.flipud(xg)
                flag2 = True

        yg = physics(xg)
        xi = model(yg, physics)

        if compute_variance:
            xout = xi

            if flag2:
                xout = torch.flipud(xout)
            if flag1:
                xout = torch.fliplr(xout)

            if max_angle > 0:
                if exact_rotations:
                    for i in range(4 - j):
                        xout = torch.rot90(xout, k=1, dims=(2, 3))
                else:
                    xout = rotate(xout, np.random.randn() * max_angle)

            if max_shift > 0:
                xout = torch.roll(xout, shifts=(-di, -dj), dims=(2, 3))

            xvar = xvar + (xout - xhat).pow(2)

        aug_error = mse(xi, xg)
        bstrap_mse[i, :] = aug_error

    if compute_variance:
        xvar = xvar / MC

    if debias:
        xhat = xmean / MC

    return xhat, bstrap_mse, xvar

def download_model(path):
    if not os.path.exists(path):
        save_dir2 = './datasets/'
        hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=path,
                        cache_dir=save_dir2, local_dir=save_dir2)

def choose_problem(problem, supervised=True, train=False):
    # choose training losses
    device = dinv.utils.get_freer_gpu()
    save_dir = f'./datasets/{problem}'
    torch.manual_seed(0)

    if not os.path.exists(f'{save_dir}/dinv_dataset0.h5'):  # download dataset
        save_dir2 = './datasets/'
        hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=f'{problem}/dinv_dataset0.h5',
                        cache_dir=save_dir2, local_dir=save_dir2)
        hf_hub_download(repo_id="jtachella/equivariant_bootstrap", filename=f'{problem}/physics0.pt',
                        cache_dir=save_dir2, local_dir=save_dir2)

    if problem == 'Tomography':
        sigma = .1
        save_dir = f'./datasets/{problem}'
        imwidth = 256

        # defined physics
        physics = dinv.physics.Tomography(img_width=imwidth, angles=40, device=device,
                                          noise_model=dinv.physics.GaussianNoise(sigma=sigma))

        # load dataset
        dataset = dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=train)

        if supervised:
            # choose a reconstruction architecture
            backbone = dinv.models.UNet(in_channels=1, out_channels=1, scales=4, # 4
                                        bias=False, batch_norm=False).to(device)
            model = dinv.models.ArtifactRemoval(backbone, pinv=True)
            losses = [dinv.loss.SupLoss()]

            if not train:
                ckp_path = f'{save_dir}/sup/ckp.pth.tar'
                download_model(f'{problem}/sup/ckp.pth.tar')
                model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
                model.eval()

        else:
            # choose a reconstruction architecture
            backbone = dinv.models.UNet(in_channels=1, out_channels=1, scales=5, # 4
                                        bias=False, batch_norm=False).to(device)
            model = dinv.models.ArtifactRemoval(backbone, pinv=True)
            losses = [dinv.loss.SureGaussianLoss(sigma=sigma, tau=.001),
                      dinv.loss.EILoss(transform=dinv.transform.Rotate(n_trans=1), weight=10, no_grad=False)]

            if not train:
                ckp_path = f'{save_dir}/rei/ckp.pth.tar' #
                download_model(f'{problem}/rei/ckp.pth.tar')
                model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
                model.eval()

        batch_size = 32

    elif problem == 'Inpainting_div2k':
        sigma = .05

        # defined physics
        physics = dinv.physics.Inpainting(mask=.5, tensor_size=(3, 256, 256), device=device,
                                          noise_model=dinv.physics.GaussianNoise(sigma=sigma))

        # load dataset
        dataset = dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=train)
        physics.load_state_dict(torch.load(f'{save_dir}/physics0.pt', map_location=device))

        # choose a reconstruction architecture
        backbone = dinv.models.UNet(in_channels=3, out_channels=3, scales=4,
                                    bias=False, batch_norm=False).to(device)

        unrolled_iter = 3
        model = dinv.unfolded.unfolded_builder(
            "HQS",
            params_algo={"stepsize": [1.0] * unrolled_iter, "g_param": [0.01] * unrolled_iter, "lambda": 1.0},
            trainable_params=["lambda", "stepsize", "g_param"],
            data_fidelity=dinv.optim.L2(),
            max_iter=unrolled_iter,
            prior=dinv.optim.PnP(denoiser=backbone),
            verbose=False,
        )
        if supervised:
            # choose a reconstruction architecture
            losses = [dinv.loss.SupLoss()]

            if not train:
                ckp_path = f'{save_dir}/sup/ckp.pth.tar'
                download_model(f'{problem}/sup/ckp.pth.tar')
                model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
                model.eval()

        else:
            # choose a reconstruction architecture
            losses = [dinv.loss.SureGaussianLoss(sigma=sigma, tau=.01),
                      dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=1), no_grad=False)]

            if not train:
                ckp_path = f'{save_dir}/rei/ckp.pth.tar'
                download_model(f'{problem}/rei/ckp.pth.tar')
                model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
                model.eval()

        batch_size = 8

    elif problem == 'CS_MNIST':
        sigma = .05

        # defined physics
        physics = dinv.physics.CompressedSensing(img_shape=(1, 28, 28), m=256, device=device,
                                          noise_model=dinv.physics.GaussianNoise(sigma=sigma))

        # load dataset
        dataset = dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=train)
        physics.load_state_dict(torch.load(f'{save_dir}/physics0.pt', map_location=device))

        norm = physics.compute_norm(torch.randn(1, 1, 28, 28).to(device))
        backbone = dinv.models.UNet(in_channels=1, out_channels=1, scales=3,
                                    bias=False, batch_norm=False).to(device)

        unrolled_iter = 4
        model = dinv.unfolded.unfolded_builder(
            "PGD",
            params_algo={"stepsize": [1.0 / norm] * unrolled_iter, "g_param": [0.01] * unrolled_iter, "lambda": 1.0},
            trainable_params=["lambda", "stepsize", "g_param"],
            data_fidelity=dinv.optim.L2(),
            max_iter=unrolled_iter,
            prior=dinv.optim.PnP(denoiser=backbone),
            verbose=False,
        )

        if supervised:
            losses = [dinv.loss.SupLoss()]

            if not train:
                ckp_path = f'{save_dir}/sup/ckp.pth.tar'
                download_model(f'{problem}/sup/ckp.pth.tar')
                model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
                model.eval()

        else:
            losses = [dinv.loss.SureGaussianLoss(sigma=sigma, tau=.01),
                      dinv.loss.EILoss(transform=dinv.transform.Shift(n_trans=4), no_grad=False)]

            if not train:
                ckp_path = f'{save_dir}/rei/ckp.pth.tar'
                download_model(f'{problem}/rei/ckp.pth.tar')
                model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
                model.eval()

        batch_size = 128

    elif problem == 'Deblur_MNIST':
        sigma = .05

        # defined physics
        kernel = torch.zeros((1, 1, 7, 7), device=device)
        kernel[0, 0, :, 3] = 1/7
        physics = dinv.physics.BlurFFT(img_size=(1, 28, 28), filter=kernel,
                                       device=device, noise_model=dinv.physics.GaussianNoise(sigma=sigma))

        # load dataset
        dataset = dinv.datasets.HDF5Dataset(path=f'{save_dir}/dinv_dataset0.h5', train=train)
        physics.load_state_dict(torch.load(f'{save_dir}/physics0.pt', map_location=device))

        norm = physics.compute_norm(torch.randn(1, 1, 28, 28).to(device))
        backbone = DnCNN(in_channels=1, out_channels=1, bias=False).to(device)

        unrolled_iter = 3
        model = dinv.unfolded.unfolded_builder(
            "PGD",
            params_algo={"stepsize": [1.0 / norm] * unrolled_iter, "g_param": [0.01] * unrolled_iter, "lambda": 1.0},
            trainable_params=["lambda", "stepsize", "g_param"],
            data_fidelity=dinv.optim.L2(),
            max_iter=unrolled_iter,
            prior=dinv.optim.PnP(denoiser=backbone),
            verbose=False,
        )

        if supervised:
            losses = [dinv.loss.SupLoss()]

            if not train:
                ckp_path = f'{save_dir}/sup/ckp.pth.tar'
                download_model(f'{problem}/sup/ckp.pth.tar')
                model.load_state_dict(torch.load(ckp_path, map_location=device)['state_dict'])
                model.eval()

        batch_size = 128

    else:
        raise Exception("problem doesn't exist")

    return physics, dataset, model, batch_size, losses, save_dir, device
