# Cloud Removal in LISS-4 Satellite Imagery using Spatiotemporal GANs

A PyTorch implementation of Spatiotemporal Generative Adversarial Networks (STGAN) for automated thick and thin cloud removal in ISRO's LISS-4 satellite imagery. This is, to our knowledge, the **first application of spatiotemporal GAN-based cloud removal to LISS-4 imagery**.

Based on the architecture from:

> Sarukkai, V., Jain, A., Uzkent, B., & Ermon, S. (2020). *Cloud Removal in Satellite Images Using Spatiotemporal Generative Networks.* IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 1796–1805. [arXiv:1912.06838](https://arxiv.org/abs/1912.06838)

---

## Overview

Optical satellite imagery is frequently occluded by clouds — roughly **two-thirds of the Earth's surface** is cloud-covered at any given moment. This project implements a **Branched U-Net generator** that ingests **three multi-temporal cloudy LISS-4 observations** acquired on different dates and reconstructs a single, cloud-free output image.

### Key features

- **Branched U-Net generator** with three parallel convolutional encoders (one per temporal observation) and a shared synthesis decoder.
- **PatchGAN discriminator** with a 70×70 receptive field for local texture realism.
- **Hybrid loss** combining conditional adversarial loss and pixel-wise L1 reconstruction loss (λ = 100).
- **Custom LISS-4 dataset** constructed from ISRO Resourcesat-2/2A archives (Green, Red, NIR bands at ~5.8 m resolution).
- **4-in-1 composite format** — each training sample is a 1024×256 image holding three cloudy frames and one ground-truth clear frame.
- **Full ablation** across loss weighting (λ = 50 vs λ = 100) and training duration (100 vs 200 epochs).

---

## Results

All experiments were conducted on a custom paired multi-temporal LISS-4 test set (50 samples).

| Model | PSNR (dB) | SSIM | MSE |
|-------|-----------|------|-----|
| Cloudy (no removal) | 8.73 | 0.5734 | 9150.67 |
| Mean filter | 10.82 | 0.5982 | 5960.54 |
| Median filter | 9.03 | 0.5460 | 8597.76 |
| Pix2Pix (single frame) | 19.20 | 0.5998 | 910.18 |
| STGAN λ=50 | 20.69 | 0.5707 | 599.19 |
| STGAN λ=100 (100 epochs) | 21.30 | 0.5949 | 543.88 |
| **STGAN λ=100 (200 epochs)** | **22.04** | **0.5973** | **485.19** |
| *Reference: Paper STGAN RGB (Sentinel-2)* | *26.00* | *0.573* | — |

**Key findings:**

- STGAN achieves the **highest PSNR (+2.84 dB over Pix2Pix, +11.22 dB over mean filtering)** and the **lowest MSE (47% reduction vs Pix2Pix)**.
- Our SSIM (0.5973) **exceeds the original paper's RGB STGAN result (0.573)** despite training on a substantially smaller dataset.
- Direct PSNR comparison to the paper is not meaningful due to domain differences (LISS-4 vs Sentinel-2 sensor, resolution, band count, dataset size).

---

## Architecture

