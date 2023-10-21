student_number = 98103867
Name = 'Mohammad'
Last_Name = 'Abolnejadian'

from Helper_codes.validator import *

python_code = extract_python("./Q1.ipynb")
with open(f'python_code_Q1_{student_number}.py', 'w') as file:
    file.write(python_code)

%pip install torchvision
%pip install numpy

from torchvision import datasets
import numpy as np
from scipy.stats import norm

train_data = datasets.MNIST('./data', train=True, download=True)
test_data  = datasets.MNIST('./data', train=False, download=True)

train_images = np.array(train_data.data)
train_labels = np.array(train_data.targets)
test_images = np.array(test_data.data)
test_labels = np.array(test_data.targets)

class Bayes:
    def train(self, train_images, train_lables):
        self.dim = 784    # 28 * 28
        self.gaussian_means = {}
        self.gaussian_stds = {}
        self.label_pros = {}
        self.label_counts = {}
        self.labels = set(train_lables)
        smoothing = 1500

        for label in self.labels:
            label_count = (train_lables == label).sum()
            self.label_counts[label] = label_count
            p = (label_count) / (len(train_lables))
            self.label_pros[label] = p
        
        for label in self.labels:
            images_for_label = train_images[train_lables == label]
            means = np.zeros((self.dim))
            stds = np.zeros((self.dim))
            for img in images_for_label:
                means += img.flatten()
                means = np.divide(means, self.label_counts[label])
                stds += np.square(img.flatten() - means)
            stds = np.divide(stds, self.label_counts[label] - 1) + smoothing

            self.gaussian_means[label] = means
            self.gaussian_stds[label] = stds
            

    def calc_accuracy(self, images, labels):
        predicted_labels = self.predict_labels(images)
        return (np.mean( labels == predicted_labels))
    
    def predict_labels(self, images):
        pred_labels = []
        for image in images:
            predicted_p = float("-inf")
            pred_label = 0
            for label in self.labels:
                p = -0.5 *(self.dim*np.log(2*np.pi) + np.sum(np.log(self.gaussian_stds[label])))\
                     -0.5 * np.sum(((image.flatten()-self.gaussian_means[label])**2) / (self.gaussian_stds[label]), 0)
                p += np.log(self.label_pros[label])
                if p > predicted_p:
                    predicted_p = p
                    pred_label = label
            pred_labels.append(pred_label)
        return np.array(pred_labels)

network = Bayes()
network.train(train_images, train_labels)

print(f"Accuracy on test data (%) : {network.calc_accuracy(test_images, test_labels) * 100:.3f}")

