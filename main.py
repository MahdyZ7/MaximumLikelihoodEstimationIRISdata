import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from scipy.stats import norm
from scipy.optimize import minimize
from pandas.plotting import scatter_matrix

def plot_distributions(feature, actual_means, actual_stds, est_means, est_stds):
    colors = ['red', 'green', 'blue']
    labels = ['Class A', 'Class B', 'Class C']
    x = np.linspace(min(actual_means)-3*max(actual_stds), max(actual_means)+3*max(actual_stds), 100)
    
    # plot actual distributions
    for i, (mean, std_dev) in enumerate(zip(actual_means, actual_stds)):
        plt.plot(x, norm.pdf(x, mean, std_dev), color=colors[i], linestyle='dashed', label=f'Actual {labels[i]}')
    
    # plot estimated distributions
    for i, (mean, std_dev) in enumerate(zip(est_means, est_stds)):
        plt.plot(x, norm.pdf(x, mean, std_dev), color=colors[i], label=f'Estimated {labels[i]}')
    plt.title(f'Estimated vs Actual distribution of  {feature}')
    plt.legend()
    plt.show()

# Define the likelihood function
def likelihood_function(params):
    mu1, sigma1, mu2, sigma2, mu3, sigma3 = params
    return -(np.log(1/3 * norm.pdf(data_col, mu1, sigma1) + 
                     1/3 * norm.pdf(data_col, mu2, sigma2) +
                     1/3 * norm.pdf(data_col, mu3, sigma3))).sum()

def calculate_error(expected_array, actual_array):
    return 100 * (np.abs(expected_array - actual_array) / actual_array)
    
def MLE(string, data, classA, classB, classC):
    print("\n\n", string)

    # Start the minimization process with some random initial guesses
    initial_params = [1, 2, 4, 5, 6, 7]
    # Set the bounds for the weights and standard deviations
    bounds = [ (None, None), (0, None), 
                (None, None), (0, None), 
                (None, None), (0, None)]

    # Run the optimizer
    result = minimize(likelihood_function, initial_params, method='SLSQP',         bounds=bounds)

    est_means = [result.x[0], result.x[2], result.x[4]]
    est_stds = [result.x[1], result.x[3], result.x[5]]
    sorted_est_means = np.sort(est_means)
    sorted_est_stds = np.sort(est_stds)
    print(f"Estimated means: {[round(num, 2) for num in sorted_est_means]}")
    print(f"Estimated standard deviations: {[round(num, 2) for num in sorted_est_stds]}")

    actual_data = [np.array(classA), np.array(classB), np.array(classC)]
    unsorted_actual_means = [np.mean(actual_data[0]), np.mean(actual_data[1]), np.mean(actual_data[2])]
    unsorted_actual_stds = [np.std(actual_data[0]), np.std(actual_data[1]), np.std(actual_data[2])]
    sorted_mean = np.sort([np.mean(actual_data[0]), np.mean(actual_data[1]), np.mean(actual_data[2])])
    sorted_std = np.sort([np.std(actual_data[0]), np.std(actual_data[1]), np.std(actual_data[2])])
    print("Actual means: %.2f %.2f %.2f" % (sorted_mean[0], sorted_mean[1], sorted_mean[2]))
    print("Actual standard deviations: %.2f %.2f %.2f" % (sorted_std[0], sorted_std[1], sorted_std[2]))



    # plot_distributions(string, unsorted_actual_means, unsorted_actual_stds, est_means, est_stds)
    percent_error = calculate_error(sorted_est_means, sorted_mean)
    print(f"Percentage error in means: {percent_error}")

    percent_error = calculate_error(sorted_est_stds, sorted_std)
    print(f"Percentage error in standard deviations: {percent_error}")

    

    
# load the iris dataset
iris = load_iris()
print(iris.keys())
print(iris['DESCR'])
print(iris['target_names'])
print(iris['feature_names'])
print(iris['data'].shape)
print(iris['target'].shape)


class_setosa = iris.target == 0
class_versicolor = iris.target == 1
class_virginica = iris.target == 2

data = iris.data
data_setosa = iris.data[class_setosa]
data_versicolor = iris.data[class_versicolor]
data_virginica = iris.data[class_virginica]


data_col = []
data_setosa_col = []
data_versicolor_col = []
data_virginica_col = []
for feature in range(data.shape[1]):
    for i in range(len(data)):
        data_col.append(data[i][feature])
    for i in range(len(data_setosa)):
        data_setosa_col.append(data_setosa[i][feature])
        data_versicolor_col.append(data_versicolor[i][feature])
        data_virginica_col.append(data_virginica[i][feature])
    MLE(iris['feature_names'][feature], data_col, data_setosa_col, data_versicolor_col, data_virginica_col)
    data_col.clear()
    data_setosa_col.clear()
    data_versicolor_col.clear()
    data_virginica_col.clear()


# plt.show()

setosa_dataframe = pd.DataFrame(iris.data[class_setosa], columns=iris.feature_names)
versicolor_dataframe = pd.DataFrame(iris.data[class_versicolor], columns=iris.feature_names)
virginica_dataframe = pd.DataFrame(iris.data[class_virginica], columns=iris.feature_names)
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Smooth Histogram using Kernel Density Estimation
# for column in iris_df.columns:
#     iris_df[column].plot.kde(legend=True)
#     plt.title(f'Smooth histogram of combined data for {column}')
#     plt.legend()
#     plt.show()
# for column in iris_df.columns:
#     setosa_dataframe[column].plot.kde(color='red',legend=True,label='Setosa')
#     versicolor_dataframe[column].plot.kde(color='green',legend=True,label='Versicolor')
#     virginica_dataframe[column].plot.kde(color='blue',legend=True,label='Virginica')
#     plt.title(f'Smooth histogram of {column}')
#     plt.legend()
#     plt.show()

# Scatter Matrix
# scatter_matrix(setosa_dataframe, alpha=0.2, figsize=(7, 7), diagonal='kde', label='setosa_dataframe')
# scatter_matrix(versicolor_dataframe, alpha=0.2, figsize=(7, 7), diagonal='kde', label='versicolor_dataframe' )
# scatter_matrix(virginica_dataframe, alpha=0.2, figsize=(7, 7), diagonal='kde', label='virginica_dataframe' )
# scatter_matrix(iris_df, alpha=0.2, figsize=(7, 7), diagonal='kde', label='all data')
# plt.show()