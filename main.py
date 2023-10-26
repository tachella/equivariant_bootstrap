import deepinv as dinv
from torch.utils.data import DataLoader
import torch
from helper_fcns import choose_problem, mse, bootstrap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

problem = 'Tomography'  # options are 'Tomography', 'Inpainting_div2k', 'CS_MNIST', 'Deblur_MNIST'
algo = 'bstrap_sup'  # options are 'bstrap_sup', 'bstrap_unsup 'naive_bstrap', 'ULA', 'diffDPIR', 'DDRM'

MC = 100  # number of Monte Carlo samples

outer_bar = False  # verbose for outer dataset loop

with torch.no_grad():
    # choose problem
    physics, dataset, model, batch_size, _, _, device = choose_problem(problem, supervised=(algo != 'bstrap_unsup'))

    num_workers = 4  # set to 0 if using small cpu, else 4
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    if algo == 'bstrap_unsup':
        max_shift = {'Tomography': 5, 'Inpainting_div2k': 10, 'CS_MNIST': 3}[problem]
        max_angle = {'Tomography': 10, 'Inpainting_div2k': 5, 'CS_MNIST': 4}[problem]
    elif algo == 'bstrap_sup':
        max_shift = {'Tomography': 0, 'Inpainting_div2k': 10, 'CS_MNIST': 3, 'Deblur_MNIST': 5}[problem]
        max_angle = {'Tomography': 8, 'Inpainting_div2k': 5, 'CS_MNIST': 0, 'Deblur_MNIST': 0}[problem]

    true_mse = np.zeros(len(dataset))
    estimated_mse = np.zeros((MC, len(dataset)))

    sigma = physics.noise_model.sigma

    start = time.time()
    k = 0
    for x, y in tqdm(dataloader, disable=not outer_bar):
        x = x.to(device)

        y = y.to(device)

        if algo == 'bstrap_sup' or algo == 'bstrap_unsup':
            xhat, samples, _ = bootstrap(x, y, physics, model, MC, max_angle, max_shift,
                                      verbose=not outer_bar)
        elif algo == 'naive_bstrap':
            xhat, samples, _ = bootstrap(x, y, physics, model, MC, 0, 0, flip=False, verbose=not outer_bar)
        elif algo == 'ULA':
            backbone = dinv.models.DnCNN(in_channels=x.shape[1], out_channels=x.shape[1],
                                                            pretrained='download_lipschitz').to(device)
            prior = dinv.optim.ScorePrior(backbone)

            norm = physics.compute_norm(x).detach().cpu().numpy()
            sigma_den = {'Tomography_finetuned':.19, 'Tomography': .178, 'Inpainting_div2k': .01, 'CS_MNIST': .035}[problem]
            thinning = 30
            burnin = .1
            model = dinv.sampling.ULA(prior, dinv.optim.L2(sigma), step_size=1/(norm/(sigma**2)+norm/sigma_den**2),
                                      sigma=sigma_den, alpha=torch.tensor(norm, device=device), verbose=not outer_bar,
                                      max_iter=int(MC*thinning/(.95-burnin)),
                                      thinning=thinning, save_chain=True, burnin_ratio=burnin, clip=(-1., 2),
                                      thresh_conv=1e-4)
            xhat, _ = model(y, physics, x_init=x)
            samples = np.zeros((MC, x.shape[0]))
            for j in range(MC):
                samples[j, :] = mse(xhat, model.get_chain()[j])

        elif algo == 'diffPIR':
            if x.shape[1] == 1:
                model = dinv.models.DRUNet(in_channels=x.shape[1], out_channels=x.shape[1]).to(device)
            else:
                model = dinv.models.DiffUNet().to(device)

            diff = dinv.sampling.DiffPIR(model, zeta=0.3, lambda_=7, data_fidelity=dinv.optim.L2(), sigma=sigma)
            model = dinv.sampling.DiffusionSampler(diff, max_iter=MC, save_chain=True, verbose=not outer_bar)
            xhat, _ = model(y, physics, x_init=physics.A_dagger(y))
            samples = np.zeros((MC, x.shape[0]))
            for j in range(MC):
                samples[j, :] = mse(xhat, model.get_chain()[j])
        elif algo == 'DDRM':
            diff = dinv.sampling.DDRM(dinv.models.DRUNet(in_channels=x.shape[1], out_channels=x.shape[1]).to(device), sigma_noise=sigma)
            model = dinv.sampling.DiffusionSampler(diff, max_iter=MC, save_chain=True, verbose=not outer_bar)
            xhat, _ = model(y, physics, x_init=physics.A_dagger(y))
            samples = np.zeros((MC, x.shape[0]))
            for j in range(MC):
                samples[j, :] = mse(xhat, model.get_chain()[j])
        else:
            raise NotImplementedError

        true_mse[k:k+x.shape[0]] = mse(x, xhat)
        estimated_mse[:, k:k+x.shape[0]] = samples
        k += x.shape[0]

    end = time.time()
    elapsed = end - start

    average_mse = np.mean(estimated_mse, axis=0)
    average_psnr = 10 * np.log10(1 / average_mse)
    true_psnr = 10 * np.log10(1 / true_mse)

    print(f'Average PSNR: {np.mean(true_psnr)}')
    print(f'Average error PSNR: {np.mean(average_psnr-true_psnr)}')

    plt.figure()
    plt.hist(average_psnr, bins=100, color='b', alpha=.5, density=True)
    plt.hist(true_psnr, bins=100, color='r', alpha=.5, density=True)
    plt.legend(['Estimated', 'True'])
    plt.show()

    percentiles = np.linspace(0.1, .99, 100)
    distance = np.sort(estimated_mse, axis=0)
    empirical_coverage = np.zeros(len(percentiles))
    for j in range(len(percentiles)):
        success = 0
        success2 = 0
        for i in range(len(dataset)):
            if true_mse[i] < distance[int(distance.shape[0] * percentiles[j]), i]:
                success += 1

        empirical_coverage[j] = success / len(dataset)

    empirical_coverage[-1] = 1.
    plt.figure()
    plt.plot(percentiles, empirical_coverage)
    plt.plot(percentiles, percentiles)
    plt.xlabel('Confidence level')
    plt.ylabel('Empirical coverage')
    plt.show()
