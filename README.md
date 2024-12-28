# Generation of Indian Sign Language Letters, Numbers, and Words

## Inference Notebook: [click here](https://)

### Official implementation for the paper [Generation of Indian Sign Language Letters, Numbers, and Words](https://ieeexplore.ieee.org/document/10721847)
#### Note: Our ProGAN code is highly inspired by Abd Elilah TAUIL from [DigitalOcean](https://blog.paperspace.com/implementation-of-progan-from-scratch/)
Contents:

**1. Description:**

Sign language generation and recognition are essential tasks for establishing effective two-way communication between individuals with hard-of-hearing and the general population. While significant advancements have been made in the recognition of sign languages from various countries, Indian Sign Language generation remains underexplored, particularly using Generative AI technologies.
Our objective is to develop a generative model capable of producing high-quality images of Indian Sign Language letters, numbers, and some popular words. The Progressive Growing of Generative Adversarial Networks (ProGAN) is known for generating high-resolution images, whereas the Self-Attention Generative Adversarial Network (SAGAN) excels at creating feature-rich images at medium resolutions. Balancing image resolution and detail is important for generating sign language images.
To address this, we designed a modified Attention-based Generative Adversarial Network (GAN) that combines the strengths of both models to produce feature-rich, high-resolution, and class-conditional Indian Sign Language images. Additionally, we are publishing a large dataset incorporating high-quality images of Indian Sign Language alphabets, numbers, and 129 words.

<p align="center">
  <img width="1080" height="450" src="https://github.com/Ajeet-kumar1/Generation-of-Indian-Sign-Language-Letters-and-Numbers/blob/main/samples/dataset_samp.png?raw=true">
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

Here in this figure sign language output corresponding to sentence "Welcome to the class where you serve." is given.

<p align="center">
  <img width="1280" height="250" src="https://github.com/Ajeet-kumar1/Generation-of-Indian-Sign-Language-Letters-and-Numbers/blob/main/samples/string9-1.png">
</p>
<p align="center">
Figure 2: Output of Sign language corresponding to "Welcome to class where you serve" with our model.
</p>

**3. Setup:**

*1. Clone the repository*

To clone type these command in your terminal line by line.

```bash
git clone https://github.com/Ajeet-kumar1/Generation-of-Indian-Sign-Language-Letters-and-Numbers.git
cd Generation-of-Indian-Sign-Language-Letters-and-Numbers
```

*2. Install Environment via Anaconda/Miniconda*
```bash
conda create -n ISLGen python=3.12.2
conda activate ISLGen
pip install -r requirements.txt

```

**3. Inference**

To perform inference, you will need the trained model weights. Ensure you have trained the model before attempting inference. Due to space limitations, we are sharing a small pre-trained model capable of generating Indian Sign Language (ISL) letters with dimensions of 64x64. Please note that this model is intended for quick testing and demonstration purposes only. You can download it from [here](www.google.com).
Once downloaded store it into your current working directory and add it's path in inference.py and run below command.

```bash
python inference.py --step 4 --num_classes 35
```
At the terminal, you can input the prompt text for generating the desired image output. This model includes 35 classes, and the generated image resolution is 64x64(step 4). When testing with your own model, please change step (step 0 = 4x4, step 1 = 8x8, step 2 = 16x16, step 3 = 32x32  and so on) and classes accordingly.

**4. Training**
The comeplete training is given in three major steps.

*1.Dataset preparation*

If you have your own dataset then add the path of dataset in ```main.py``` file else if you want to use our dataset then follow given steps:

(i) Download the dataset videos from [this link](www.my_data.com)

(ii) Now in the ```extract.py``` file in this line ```cam = cv2.VideoCapture("./my_video/class_0.mp4")``` add path of a class of video. And also change the root_path to store images corresponding to this class.

(iii) Run the following command ```python3 extract.py```

(iii) Repeat the step (ii) and (iii) for remaining classes. Now you have dataset in folder by name ```data```.

*2. Hyperparamater tuning*

While training you need to keep this changes based on your requirements.

*3. Train the model*

To train the model run the following command.

```
python3 main.py
```

**5. Evaluation**

For the evaliuation a quite.


