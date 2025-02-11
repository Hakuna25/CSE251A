import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import random

class OneNN:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        # Store training data and labels
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        # Initialize a list to store predictions
        predictions = []
        
        # Loop over each point in the test set
        for x_test in X:
            # Compute distances to all training points
            distances = np.linalg.norm(self.X_train - x_test, axis=1)
            
            # Find the index of the closest point (smallest distance)
            closest_index = np.argmin(distances)
            
            # Get the label of the closest point
            predictions.append(self.y_train[closest_index])
        
        return np.array(predictions)
    def score(self, X_test, y_test):
        """Computes the accuracy of the classifier on the test set."""
        y_pred = self.predict(X_test)  # Get predictions
        accuracy = np.mean(y_pred == y_test)  # Compare with ground truth
        return accuracy
    
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def random_selection(train_dataset, M):
    X = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
    prototype = random.sample(range(0, X.shape[0]),M)
    return prototype

def cnn_selection(train_dataset, M):
    X = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
    # breakpoint()
    y = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    prototype = []
    prototype.append(0)
    rand = random.sample(range(0,X.shape[0]),X.shape[0])
    # breakpoint()
    additions = True
    while additions == True:
        additions = False
        for i in range(X.shape[0]):
            if len(prototype) >= M:
                return prototype
            nn_model = OneNN() # KNeighborsClassifier(n_neighbors = 1)
            nn_model.fit(X[prototype],y[prototype])
            # if nn_model.predict(X[i].reshape(1,-1)) != y[rand[i]]:
            if nn_model.predict(X[rand[i]].reshape(1,-1)) != y[rand[i]]:
                prototype.append(rand[i])
                additions = True
            if i % 1000 == 0:
                print(f"cnn_selection:{i}, len(prototype):{len(prototype)}")
    return prototype

def kmeans_selection(train_dataset, M):
    X_train = np.array([train_dataset[i][0].numpy() for i in range(len(train_dataset))])
    # breakpoint()
    y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
    #prototypes holds the indexes of the training set that will be used as prototypes
    prototype = []
    classes = len(np.unique(y_train))
    num_clusters = int(M/classes)
    x_index = [ [] for _ in range(classes) ]
    for i in range(len(y_train)):
        x_index[y_train[i]].append(i)
    iter = 0
    for x in x_index:
        cc = KMeans(n_clusters = num_clusters).fit(X_train[x]).cluster_centers_
        pt, _ = pairwise_distances_argmin_min(cc,X_train)
        prototype.append(pt)
    prototype = np.concatenate(prototype).ravel()
    return prototype

def evaluate(train_subset, prototype, test_dataset):
    X_train = np.array([train_subset[i][0].numpy() for i in range(len(train_subset))])
    y_train = np.array([train_subset[i][1] for i in range(len(train_subset))])
    X = X_train[prototype]
    y = y_train[prototype]
    X_test = np.array([test_dataset[i][0].numpy() for i in range(len(test_dataset))])
    y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
    knn = OneNN() # KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)
    accuracy = knn.score(X_test, y_test)
    return accuracy

if __name__ == "__main__":
    train_dataset, test_dataset = load_mnist()
    M_values = [1000, 5000, 10000]
    seed = [0, 1, 42]
    num_trials = 3
    for M in M_values:
        random_accuracies = []
        cnn_accuracies = []
        kmeans_accuracies = []
        for i in range(num_trials):
            random.seed(seed[i])
            print(f"This is trial:{i}!!!")
            # random_prototype = random_selection(train_dataset, M)
            # print("Random_subset finish!")
            # random_acc = evaluate(train_dataset, random_prototype, test_dataset)
            
            # cnn_prototype = cnn_selection(train_dataset, M)
            # print("Cnn_prototype finish!")
           
            # cnn_acc = evaluate(train_dataset, cnn_prototype, test_dataset)
            # random_accuracies.append(random_acc)
            # cnn_accuracies.append(cnn_acc)
            # print(f"random_acc:{random_acc}, cnn_acc:{cnn_acc}")
            kmeans_prototype = kmeans_selection(train_dataset, M)
            print("Kmeans_prototype finish!")
            kmeans_acc = evaluate(train_dataset, kmeans_prototype, test_dataset)
            kmeans_accuracies.append(kmeans_acc)
            print(f"kmeans_acc:{kmeans_acc}, kmeans_acc:{kmeans_acc}")
            
        print(f"M={M}")
        # print(f"Random Selection: {np.mean(random_accuracies):.4f} ± {np.std(random_accuracies):.4f}")
        # print(f"CNN Selection: {np.mean(cnn_accuracies):.4f} ± {np.std(cnn_accuracies):.4f}")
        print(f"Kmeans Selection: {np.mean(kmeans_accuracies):.4f} ± {np.std(kmeans_accuracies):.4f}")
    






