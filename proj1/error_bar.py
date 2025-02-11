import numpy as np
import matplotlib.pyplot as plt

# 定义 M 值
M_values = np.array([1000, 5000, 10000])

# CNN
method_1_mean = [0.8931, 0.9329, 0.9443] 
method_1_std = [0.0011, 0.0011, 0.0011]  
# Kmeans
method_2_mean = [0.9296, 0.9497, 0.9534] 
method_2_std = [0.0018, 0.0013, 0.0011] 
# Random
method_3_mean = [0.8843, 0.9345, 0.9479] 
method_3_std = [0.0035, 0.0005, 0.0017]  

plt.figure(figsize=(8, 6))

plt.errorbar(M_values, method_1_mean, yerr=method_1_std, fmt='o-', capsize=5, label="CNN", color='r')
plt.errorbar(M_values, method_2_mean, yerr=method_2_std, fmt='s-', capsize=5, label="Kmeans", color='pink')
plt.errorbar(M_values, method_3_mean, yerr=method_3_std, fmt='d-', capsize=5, label="Random", color='purple')

plt.xlabel("Number of Prototypes (M)")
plt.ylabel("Classification Accuracy")
plt.title("Comparison of Different Methods with Error Bars")
plt.legend()
plt.grid(True)
plt.savefig('error_bar.png')
plt.show()

