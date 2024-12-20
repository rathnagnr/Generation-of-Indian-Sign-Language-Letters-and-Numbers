# Generation of Indian Sign Language Letters, Numbers, and Words

## Inference Notebook: [click here](https://)

### Official implementation for the paper [Generation of Indian Sign Language Letters, Numbers, and Words](https://ieeexplore.ieee.org/document/10721847)
#### Note: Our ProGAN code is highly inspired by Abd Elilah TAUIL from [DigitalOcean](https://blog.paperspace.com/implementation-of-progan-from-scratch/)
Contents:

**1. Description:**

Sign language generation and recognition are essential tasks for establishing effective two-way communication between individuals with hard-of-hearing and the general population. While significant advancements have been made in the recognition of sign languages from various countries, Indian Sign Language generation remains underexplored, particularly using Generative AI technologies.
Our objective is to develop a generative model capable of producing high-quality images of Indian Sign Language letters, numbers, and some popular words. The Progressive Growing of Generative Adversarial Networks (ProGAN) is known for generating high-resolution images, whereas the Self-Attention Generative Adversarial Network (SAGAN) excels at creating feature-rich images at medium resolutions. Balancing image resolution and detail is important for generating sign language images.
To address this, we designed a modified Attention-based Generative Adversarial Network (GAN) that combines the strengths of both models to produce feature-rich, high-resolution, and class-conditional Indian Sign Language images. Our approach outperforms ProGAN, achieving significant improvements in Inception Score (IS) and Frechet Inception Distance (FID), with increases of 3.2 and 30.12, respectively. Additionally, we are publishing a large dataset incorporating high-quality images of Indian Sign Language alphabets, numbers, and 129 words.

**2. Methodology:**
Our proposed architecture consists of two main components: a generator and a discriminator (critic) network, as illustrated in below Figure 1. The spatial resolution of the layers in both networks changes in multiple stages. In the generator, each stage’s layers double in resolution, starting at 8x8 to capture large-scale structures and increasing to capture finer details.
The generator takes two inputs in the first stage: a Gaussian latent noise and a class label. After embedding the class label, it is concatenated with the latent noise, passing through the
progressively growing generator networks to produce detailed images. In general, images start possessing finer detail after a 64x64 resolution. At the 64x64
resolution stage, in between two convolution layers, a Self-attention layer is added. In Figure 1, the first Self-attention layer may appear before the 64x64 resolution layer, but it
is present between two convolution layers of 64x64. Self-attention allows the convolution layer to focus on specific parts of the image regions that are most relevant in the
images with their conditions. Another Self-attention layer is applied between two convolution layers of resolution 128x128.The critic network, whose structure is almost a mirror
image of the generator network receives the generator’s output and the class label while training. 
**3. Examples:**

**3. Installation:**

*1. Requirements Installation*

*2. Inference*

*3. Dataset ceation*

*4. Training*

*5. Evaluation*


# This repository under progress we are updating the repository and adding code.

# first change by local. Now second change by local.
