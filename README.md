# Zero-Shot Text-to-Image Generation for Housing Floor Plans

## Contributors:
* Shannon Phu, shannon.phu@sjsu.edu, San Jose State University
* Shiv Kumar Ganesh, shivkumar.ganesh@sjsu.edu, San Jose State University
* Kumuda BG Murthy, kumuda.benakanahalliguruprasadamurt@sjsu.edu, San Jose State University
* Raghava D Urs, raghavadevaraje.urs@sjsu.edu, San Jose State University

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