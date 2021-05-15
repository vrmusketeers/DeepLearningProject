# Zero-Shot Text-to-Image Generation for Housing Floor Plans

## Summary
Our data relies on the [CubiCasa5k housing floorplan dataset](https://zenodo.org/record/2613548). We parsed the SVG data to convert the images to annotate rooms types with a particular room color and to generate a textual description of the house. We then trained a discrete variational autoencoder (VAE) to learn the image representation embeddings for the floorplan images. The VAE aims to reduce the reconstruction loss and KL divergence loss. Afterwards, we trained a DALL-E model which accepts the VAE and our text data to learn to generate an image from the input text. Our DALL-E model aims to reduce the cross entropy loss weighted between text and image. Our tech stack uses PyTorch, the dalle-pytorch package, Weights & Biases for MLOps, and Streamlit for our production inference website.

### Report
[Report](https://github.com/vrmusketeers/DeepLearningProject/blob/main/documentation/Report-Text-to-Image%20Generation%20for%20Housing%20Floor%20Plans.pdf)

### Slidedeck
[Slidedeck](https://github.com/vrmusketeers/DeepLearningProject/blob/main/documentation/Slides%20-%20Zero-Shot%20Text-to-Image%20Generation%20for%20Housing%20Floor%20Plans.pdf)

### Presentation
Video Presentaion [Click here](https://www.youtube.com/watch?v=v4fOhLyr6Hg)

## Contributors:
* Shannon Phu, shannon.phu@sjsu.edu, San Jose State University
* Shiv Kumar Ganesh, shivkumar.ganesh@sjsu.edu, San Jose State University
* Kumuda BG Murthy, kumuda.benakanahalliguruprasadamurt@sjsu.edu, San Jose State University
* Raghava D Urs, raghavadevaraje.urs@sjsu.edu, San Jose State University

| Contributor       | Contributions                                                               |
|-------------------|-----------------------------------------------------------------------------|
| Shannon Phu       | trained VAE/DALL-E models                                                   |
| Shiv Kumar Ganesh | Streamlit inference website, inference pipeline, data collection/processing |
| Kumuda BG Murthy  | data collection/processing, MLOps                                           |
| Raghava D Urs     | experimented with StackGAN model, inference pipeline, Expiremented with AWS SageMaker                      |

## Inference Website Demo
https://share.streamlit.io/vrmusketeers/deeplearningproject/main/streamlitapp/streamlit.py

## Code:
1. Generate Data/Data Preparation Notebook [[ Github Code](https://github.com/vrmusketeers/DeepLearningProject/blob/main/notebooks/Generate_Data.ipynb) | [Colab](https://colab.research.google.com/github/vrmusketeers/DeepLearningProject/blob/main/notebooks/Generate_Data.ipynb) ]
    * data preparation
2. Training Notebook [ [Github Code](https://github.com/vrmusketeers/DeepLearningProject/blob/main/notebooks/Train_VAE_and_DALLE.ipynb) | [Colab](https://colab.research.google.com/github/vrmusketeers/DeepLearningProject/blob/main/notebooks/Train_VAE_and_DALLE.ipynb) ]
    * train/test split
    * image data augmentation
    * VAE training
    * DALL-E training
    * MLOps (with WandB)
        * training pipeline tracking
        * training pipeline run comparisons
        * model versioning
        * model repository
        * evaluation dashboard
3. Inference Notebook [ [Github Code](https://github.com/vrmusketeers/DeepLearningProject/blob/main/notebooks/Inference.ipynb) | [Colab](https://colab.research.google.com/github/vrmusketeers/DeepLearningProject/blob/main/notebooks/Inference.ipynb) ]
    * notebook inference code in Colab
4. Website Inference Demo [ [Github Code](https://github.com/vrmusketeers/DeepLearningProject/blob/main/streamlitapp/streamlit.py) | [Demo Link](https://share.streamlit.io/vrmusketeers/deeplearningproject/main/streamlitapp/streamlit.py) ]
    * online website inference in real-time

## Documentation
1. Report on project implementation [Click here](https://github.com/vrmusketeers/DeepLearningProject/blob/main/documentation/Report-Text-to-Image%20Generation%20for%20Housing%20Floor%20Plans.pdf)
2. Slide deck [Click here](https://github.com/vrmusketeers/DeepLearningProject/blob/main/documentation/Slides%20-%20Zero-Shot%20Text-to-Image%20Generation%20for%20Housing%20Floor%20Plans.pdf)

