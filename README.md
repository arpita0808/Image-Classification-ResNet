# Image Classification using ResNet18

##  Overview
This project implements an image classification system using deep learning. A pretrained ResNet18 model is used with transfer learning to classify images from the CIFAR-10 dataset.

---

##  Features
- Transfer Learning using ResNet18
- Data Augmentation
- Training & Validation Analysis
- CNN vs ResNet Comparison
- User Input Image Prediction
- Model Saving & Loading

---

##  Dataset
CIFAR-10 dataset with 10 classes:
- airplane, car, bird, cat, deer, dog, frog, horse, ship, truck

---

## ⚙️ Technologies Used
- Python
- PyTorch
- Torchvision
- Matplotlib

---

##  How to Run

### 1. Install dependencies
```bash
pip install torch torchvision matplotlib
```

### 2. Run the notebook
```bash
DEEP_LEARNING.ipynb
```

### 3. Predict using image
```python
predict_image("test.jpg")
```

---

##  Results
- ResNet18 achieved higher accuracy compared to the custom CNN model  
- Transfer learning improved performance and reduced training time  
- The model shows lower confidence on real-world images due to dataset differences

##  Model Comparison Graph

The graph below shows the comparison between the custom CNN model and the pretrained ResNet18 model.  
It clearly illustrates that ResNet18 achieves higher validation accuracy due to transfer learning and a deeper architecture.

<img width="1237" height="704" alt="image" src="https://github.com/user-attachments/assets/efe4e435-8df9-454a-8618-6513a080c1cb" />

##  Download Trained Model

Due to GitHub file size limitations, the trained model is hosted externally.

https://drive.google.com/file/d/1A-OxV4nY28LTS4my8P3hNkYwRqFM1UmJ/view?usp=drive_link

## 🧠 Conclusion

This project demonstrates a complete deep learning pipeline for image classification using transfer learning.  
The pretrained ResNet18 model significantly outperforms a custom CNN, highlighting the effectiveness of using pretrained architectures.

The system supports training, evaluation, model comparison, and real-time prediction on user-provided images.  
Although the model performs well on CIFAR-10-like images, its confidence decreases on real-world images due to differences in resolution and data distribution.

Overall, this project successfully showcases practical implementation of deep learning techniques and model deployment.
