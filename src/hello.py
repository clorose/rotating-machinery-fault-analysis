import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.datasets as datasets
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def test_libraries():
    print("Testing NumPy...")
    arr = np.random.randn(5, 5)
    print("NumPy array shape:", arr.shape)
    
    print("\nTesting Pandas...")
    df = pd.DataFrame(arr, columns=['A', 'B', 'C', 'D', 'E'])
    print("Pandas DataFrame head:")
    print(df.head())
    
    print("\nTesting SciPy...")
    z_score = stats.zscore(arr.flatten())
    print("SciPy z-score shape:", z_score.shape)
    
    print("\nTesting Scikit-learn...")
    iris = datasets.load_iris()
    print("Iris dataset shape:", iris.data.shape)
    
    print("\nTesting PyTorch...")
    tensor = torch.from_numpy(arr)
    print("PyTorch tensor shape:", tensor.shape)
    
    print("\nTesting Matplotlib & Seaborn...")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig('/app/runs/test_plot.png')
    plt.close()
    
    print("\nAll libraries imported and tested successfully!")
    print("A correlation heatmap has been saved as 'test_plot.png'")

if __name__ == "__main__":
    test_libraries()