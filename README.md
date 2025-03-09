# DeepFake Detector: EfficientNet vs ResNet  

This project detects DeepFake images using **EfficientNet-B0** and **ResNet18**. The models are trained on real and fake images, and a **Streamlit web app** allows users to upload images and compare predictions between the two models.  

---

## Features  

- **DeepFake detection** using ResNet18 and EfficientNet-B0  
- **User-friendly** Streamlit web app for easy predictions  
- **Comparison of models** with accuracy and loss metrics  
- **Supports custom image uploads** for testing  

---

## Dataset  

The dataset consists of:  

- **Fake images** (`/fake/`)  
- **Real images** (`/real/`)  
- **`data.csv`** containing metadata  

**Dataset Source**: [Kaggle](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces?resource=download)  

---

## Future Improvements

- Use more diverse datasets for better generalization
- Improve model interpretability with Grad-CAM visualization

---
