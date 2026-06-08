OBJECTIVE:
To build a neural network model to classify Fashion-MNIST dataset images into their respective clothing categories

DATASET:
The Fashion-MNIST dataset consists of grayscale images of various clothing belonging to 10 different clothing categories.size of images = 28*28 pixels

APPROACH
1.Loaded and preprocessed the dataset.
2.Normalizing pixel values in range[0,1]
3.Convert this data into PyTorch tensors.
4.Training the model on train_dataset using CrossEntropyLoss and an optimizer.
5.Evaluate the model performance using validation accuracy and loss.
6.Generated predictions for test dataset and saved them as submission.csv.

FILES:
-> MLtasksp.ipynb: training and evaluation colab notebook
-> fashion_weights.pkl: the saved weights
-> submission.csv: test set predictions

RESULTS:
model trained for 20 epochs.
validation Accuracy=86.43

(RUNNING THE DATASET:
the dataset files are manually uploaded, so ensure to upload them before running)
