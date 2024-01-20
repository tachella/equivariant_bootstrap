# Equivariant Bootstrapping for Uncertainty Quantification in Imaging Inverse Problems

Code of the paper
["Learning to reconstruct signals from binary measurements alone"](https://arxiv.org/abs/2303.08691) by [Julian Tachella](https://tachella.github.io/)
and [Marcelo Pereyra](https://www.macs.hw.ac.uk/~mp71/about.html), 

AISTATS 2024, **oral presentation**.

We use the [deepinv library](https://deepinv.github.io/deepinv/)
for most of the code.


![image info](./bootstrap_schematic.png)


# Method Description

The equivariant bootstrap method is a simple resampling procedure which produces well-calibrated uncertainty intervals 
for any reconstruction network (i.e., trained in a supervised or self-supervised manner). The method outperforms other 
SOTA diffusion-based methods in terms of coverage, speed and test PSNR.

Assuming that the forward process associated with the inverse problem is described as 

$$
y = \mathcal{P}(Ax)
$$

where $x$ is the image that we want to recover, $y$ are the observed measurements,
$A$ is a linear operator and $\mathcal{P}$ is a stochastic process modelling the noise 
(e.g., the addition of Gaussian or Poisson noise). For a reconstruction network $\hat{x}(y)$ with input measurement $y$, the method generates new samples as

$$
x^{(i)} = T_g^{-1}\hat{x}\left(\mathcal{P}(AT_g\hat{x}(y))\right)
$$

for $i=1,\dots,N$ where $T_g$ is a group transformation (e.g., rotation, translation, etc.). These samples are then used to compute 
confidence intervals of the MSE and/or pixel-wise variants (or any other statistic of interest). Check the paper for more details!

# Getting Started
1. Clone the repository
2. Install the latest version of [deepinv](https://deepinv.github.io/) if you don't have it already
```
pip install git+https://github.com/deepinv/deepinv.git#egg=deepinv
```
3. Install huggingface_hub if you don't have it already
```
pip install huggingface_hub
```
4. Run the `main.py` file to run the proposed equivariant bootstrapping technique for uncertainty quantification, and
the other competing methods (ULA, diffPIR and DDRM). The script downloads automatically the datasets and pretrained models from the [deepinv library](https://deepinv.github.io/deepinv/)
and the HuggingFace repo [jtachella/equivariant_bootstrap](https://huggingface.co/jtachella/equivariant_bootstrap).

# Results
The following figure shows the coverage of the proposed method (Equivariant Bootstrap) and the other competing methods
(see the paper for more details). The proposed method significantly outperforms the other methods in terms of coverage. 

![image info](./coverage.png)

The following table shows the average test PSNR of the proposed method (Equivariant Bootstrap) and the other competing methods:

|                        |       DDRM      |      DiffPIR     |        ULA       |     Proposed self-sup.    |       Proposed sup.      |
|------------------------|:---------------:|:----------------:|:----------------:|:-------------------------:|:------------------------:|
| C. Sensing (MNIST)     |         -       |         -        | $28.54\pm 2.25$  | $\mathbf{34.11 \pm 2.09}$ | $33.9 \pm 2.32$          |
| Inpainting (DIV2K)     | $32.27\pm 3.95$ |  $30.51\pm 3.74$ | $30.52 \pm 3.35$ | $31.56 \pm 4.12$          | $\mathbf{32.47\pm 3.87}$ |
| Tomography (LIDC-IDRI) |        -        | $37.02 \pm 0.79$ | $35.85 \pm 0.54$ | $37.38 \pm 0.65$          | $\mathbf{41.03\pm 0.91}$ |

# Citation
```
@article{tachella2023bootstrap,
  title={Equivariant Bootstrapping for Uncertainty Quantification in Imaging Inverse Problems},
  author={Tachella, Julian and Pereyra, Marcelo},
  journal={arXiv preprint arXiv:2310.11838},
  year={2023}
}
```
