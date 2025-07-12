# 🐱🐶cat_vs_dog_classifier
📌<b> Objective:</b>
<br>The goal of this project is to build a machine learning model that classifies images of cats and dogs. Using Convolutional Neural Networks (CNNs), the model learns visual features to accurately differentiate between the two categories.

🧠 <b>Project Overview:</b>
<br>Built an image classification model using deep learning (CNN).<br>
Used the Cats vs Dogs dataset from Kaggle.<br>
Trained the model using TensorFlow/Keras for high accuracy.<br>
Evaluated performance using accuracy, confusion matrix, and sample predictions.

🧾<b> Dataset:</b>
<br>Source: Kaggle - Dogs vs. Cats<br>
Contains 25,000 images:<br>
12,500 cat images<br>
12,500 dog images

🛠️<b> Technologies Used:</b>
<br>Python<br>
TensorFlow / Keras<br>
OpenCV / PIL<br>
Matplotlib / Seaborn<br>
NumPy / Pandas

🧪 <b>Model Workflow:</b>
<br>1. <b>Data Preprocessing:</b>
<br>Resized images to 128x128 or 150x150.<br>
Normalized pixel values.<br>
Applied data augmentation (rotation, flip, zoom) to improve generalization.

<b>2. Model Building:</b>
<br>Convolutional layers for feature extraction.<br>
Pooling layers for dimensionality reduction.<br>
Dense layers for classification (Softmax/Sigmoid).

<b>3. Training:</b>
<br>Loss Function: Binary Crossentropy<br>
Optimizer: Adam<br>
Metrics: Accuracy

<b>4. Evaluation:</b>
<br>Accuracy score on validation/test set<br>
Confusion matrix and visual predictions

📈<b> Results:</b>
<br>Achieved training accuracy of ~97% and validation accuracy of ~92%.<br>
Model performs well in distinguishing cats from dogs in unseen images.

✅ <b>Conclusion:</b>
<br>This project demonstrates how CNNs can be effectively used for binary image classification. With proper preprocessing, augmentation, and training, the model generalizes well to new data. It can be further enhanced by tuning hyperparameters or using transfer learning with pre-trained models like VGG16 or ResNet.
