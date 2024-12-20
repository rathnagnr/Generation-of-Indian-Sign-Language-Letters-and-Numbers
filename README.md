# Generation of Indian Sign Language Letters, Numbers, and Words

## Inference Notebook: [click here](https://)

### Official implementation for the paper [Generation of Indian Sign Language Letters, Numbers, and Words](https://ieeexplore.ieee.org/document/10721847)
#### Note: Our ProGAN code is highly inspired by Abd Elilah TAUIL from [DigitalOcean](https://blog.paperspace.com/implementation-of-progan-from-scratch/)
Contents:

**1. Description:**

Sign language generation and recognition are essential tasks for establishing effective two-way communication between individuals with hard-of-hearing and the general population.
Our objective is to develop a generative model capable of producing high-quality images of Indian Sign Language letters, numbers, and some popular words. The Progressive Growing of Generative Adversarial Networks (ProGAN) is known for generating high-resolution images, whereas the Self-Attention Generative Adversarial Network (SAGAN) excels at creating feature-rich images at medium resolutions. Balancing image resolution and detail is important for generating sign language images.
To address this, we designed a modified Attention-based Generative Adversarial Network (GAN) that combines the strengths of both models to produce feature-rich, high-resolution, and class-conditional Indian Sign Language images. Additionally, we are publishing a large dataset incorporating high-quality images of Indian Sign Language alphabets, numbers, and 129 words.

<p align="center">
  <img width="1080" height="550" src="https://github.com/Ajeet-kumar1/Generation-of-Indian-Sign-Language-Letters-and-Numbers/blob/main/samples/dataset_samp.png?raw=true">
</p>
<p align="center">
Figure 1: Some samples of our dataset
</p>

**2. Methodology:**

Our proposed architecture consists of two main components: a generator and a discriminator network, as illustrated in below Figure 2.
The generator takes two inputs in the first stage: a Gaussian latent noise and a class label. At the 64x64
resolution stage, in between two convolution layers, a Self-attention layer is added. Another Self-attention layer is applied between two convolution layers of resolution 128x128.The critic network, whose structure is almost a mirror
image of the generator network receives the generator’s output and the class label while training. 
<p align="center">
  <img width="900" height="800" src="https://github.com/Ajeet-kumar1/Generation-of-Indian-Sign-Language-Letters-and-Numbers/blob/main/samples/architect.png?raw=true">
</p>
<p align="center">
Figure 2: Our proposed architecture.
</p>

**3. Examples:**

**3. Installation:**

*1. Requirements Installation*

*2. Inference*

*3. Dataset ceation*

*4. Training*

*5. Evaluation*


# This repository under progress we are updating the repository and adding code.

# first change by local. Now second change by local.
