# Biometric Emotion Prediction System (CNN)

![Python](https://img.shields.io/badge/Python-Data_Science-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Multimodal_Fusion-success)

Convolutional Neural Network (CNN) architecture designed to predict human emotions by processing and fusing multimodal biometric signals.

## 📌 Project Overview
The system utilizes deep learning to interpret physiological data and classify emotional states[cite: 6]. By leveraging PyTorch, the project explores different data integration strategies to maximize predictive accuracy from heterogeneous sensor inputs[cite: 6].

## ⚙️ Key Features & Architecture
* **Multimodal Data Processing:** Designed and trained a CNN to predict human emotions based on complex biometric signals, including ECG and sweat levels[cite: 6].
* **Sensor Fusion Techniques:** Implemented advanced machine learning architectures, developing and comparing both Early Fusion and Late Fusion techniques to effectively integrate diverse data sources[cite: 6].
* **Model Interpretability:** Executed graph-based feature analysis to interpret the neural network's behavior, identifying which specific biometric inputs most significantly influenced the final predictive output[cite: 6].

## 🚀 How to Run
1. Clone the repository: `git clone https://github.com/alex-martinelli/biometric-emotion-cnn.git`
2. Create a virtual environment and install the dependencies: `pip install -r requirements.txt`
3. Ensure the required biometric dataset is placed in the `data/` directory.
4. Run the training script: `python train.py`
5. Evaluate the model performance and generate the feature analysis graphs: `python evaluate.py`
## 📄 Research Paper & Documentation
For a deep dive into the network architecture, hyperparameter tuning, and the mathematical reasoning behind the multimodal fusion strategies, please refer to the official project report:
* [**Emotion Recognition from Biometric Signals (PDF)**](./docs/Emotion_Recognition_Research_Report.pdf)

The paper includes comprehensive SHAP interpretability analyses, illustrating how the model evaluates localized ECG spikes versus diffused EDA state-validators to resolve high-arousal emotional ambiguities.
