# HistologyNet: Revolutionizing Biomedical Image Segmentation with Self-Supervised Learning

## Overview
HistologyNet leverages state-of-the-art self-supervised learning techniques to enable efficient binary segmentation of histology images. By utilizing the Vision Transformer (ViT) encoder and the Segment Anything Model (SAM) architecture, our solution minimizes the reliance on extensive manual annotations, offering a faster, cost-effective alternative for biomedical image segmentation.

### Key Features of Our Solution
1. **Vision Transformer (ViT) Encoder**:  
   Utilized for generating robust image embeddings, ensuring high-quality segmentation.

2. **Decoder Architecture**:  
   Decodes the embeddings into the original image, forming the basis for segmentation tasks.

3. **Fine-Tuning with SAM**:  
   Incorporates ViT Encoder and Decoder from the SAM framework, further refined using labeled data for enhanced performance.

4. **Data Preprocessing for Enhanced Results**:  
   We applied techniques from the RANDstainNA paper for stain normalization, significantly improving segmentation accuracy.

---

## How to Train the Model

### Prerequisites
Before training, ensure the following dependencies are installed:

```bash
pip install scipy
pip install opencv-python
pip install torch
pip install matplotlib
pip install datasets
pip install transformers
pip install tqdm
```


## Training Instructions

### Set Up Dataset and Model Paths
Update the dataset paths (images and masks) and specify the folder for saving trained models (e.g., `./models`) in the provided scripts.

### Run the Training
Execute the training pipeline using the `run.py` and `main.sh` scripts on your local environment or Google Colab.

### Pre-Trained Models
For convenience, we have shared our pre-trained models. You can download them using the link below and directly use them for segmentation tasks.  
[Download Pre-Trained Models](https://drive.google.com/drive/folders/1fl-3Yg_t5HiIawgrhZf3glpu2fw-Yl_z?usp=sharing)

---

## Additional Resources

### Presentation Video
Watch our detailed presentation to better understand our approach and implementation:  
[View Presentation Video](https://drive.google.com/drive/folders/1LwZcLR13pL8_ZncOeKVZSLHrfpRO4jA_?usp=sharing)

---

## Folder Structure and Instructions
Each folder within the repository contains a `README.md` file, providing detailed usage instructions and explanations for the respective scripts and components.

---

## Highlights of Our Approach
- **Fine-Tuning SAM**: A cutting-edge architecture tailored for histology images.
- **Stain Normalization**: Preprocessing techniques to handle variations in histology data effectively.
- **User-Friendly Webpage**: Developed an intuitive webpage where users can upload histology images and receive segmented masks, showcasing the real-world applicability of our solution.

We believe this solution demonstrates the potential of self-supervised learning to revolutionize biomedical image analysis.
