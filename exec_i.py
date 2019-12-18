# AUTHOR:      TOMOKI KOIKE
# DESCRIPTION: THIS FILE IS AIMED TO EXECUTE THE KMEAN.PY AND EXPERIMENT CLUSTERING. THIS PARTICULAR PYTHON WILL
#              CLUSTER THE NUMBERS 2, 4, 6, AND 7.
###

# Importing the class from python file kmean.py
from kmean import MyKmeans
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Main function
def main():
    # Defining an object for the class MyKmeans
    km = MyKmeans()
    # Parsing the data file and acquiring a dataframe
    parsedData = km.readData('digits-embedding.csv')
    # Defining the numbers you need within the datafile to create a subset
    sub_nums = [2, 4, 6, 7]
    # Creating the subset
    subset = createSubset(parsedData, sub_nums)

    #--- The Tasks ---#
    # (1) Visualizing the images based on the 2D features
    visualizeData(subset, sub_nums)
    # (2) Cluster the data with different values of K ∈ [2,4,8,16]. For each K
    # repeat the experiment for 5 different times each with random centroids and
    # calculate the average Silhouette Coefficient (SC) of each K after the 5 trials
    K = [2, 4, 8, 16]
    # >> K = 2
    SC_avg_K2 = iterClustering(subset, parsedData, 5, 50, K[0])
    # >> K = 4
    SC_avg_K4 = iterClustering(subset, parsedData, 5, 50, K[1])
    # # >> K = 8
    SC_avg_K8 = iterClustering(subset, parsedData, 5, 50, K[2])
    # # >> K = 16
    SC_avg_K16 = iterClustering(subset, parsedData, 5, 50, K[3])

    # Plotting results
    plotAvgSC(K, [SC_avg_K2, SC_avg_K4, SC_avg_K8, SC_avg_K16])


# Function to create the subsets of the parsed Data
def createSubset(parsedData, sub_nums):
    # Conducting boolean indexing on the dataframe to obtain subsets
    a = 'Digit Label'
    # Initializing the subset dataframe
    data_sub = parsedData[parsedData[a] == sub_nums[0]]
    # Looping to concatenate the other digits necessary for the subset
    for i in range(len(sub_nums)-1):
        temp = parsedData[parsedData[a] == sub_nums[i+1]]
        data_sub = pd.concat([data_sub, temp])
    return data_sub.reset_index()

# Function to create a scattered visual plot for the parsed data
def visualizeData(subset, sub_nums):
    # Initializing a list of colors for the plots
    colors = ['blue', 'orange', 'green', 'red', 'purple',
              'brown', 'pink', 'grey', 'olive', 'cyan']
    # Looping through the parsed data to plot for each number
    for i in sub_nums:
        numData = subset[subset['Digit Label'] == i]
        xval = list(numData['x'])
        yval = list(numData['y'])
        # Plotting
        plt.figure()
        plt.scatter(xval, yval, c=colors[i-1])
        plt.title('Visualized Image of the Digit {0} using \n2D Features - By: Tomoki Koike'.format(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(b=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        plt.savefig('visualImage_{0}_1.png'.format(i))
    return

# Function to reset the index for the obtain cluster list
def resetClusterIndex(cluster):
    cluster_df = pd.DataFrame(cluster)
    cluster = cluster_df.reset_index()
    return np.array(cluster.T)[:, 1]

# Function to calculate the average SC
def calAvgSC(clusters, parsedData):
    km = MyKmeans()
    # Intialize SC value
    SC = 0
    # Looping to obtain average SC value for all trials
    for x in range(clusters.shape[0]):
        temp = km.calculateSC(clusters[x], parsedData)
        SC += temp
    SC_avg = SC/clusters.shape[0]
    return SC_avg

# Function to iterate the operation of obtaining the SC_avg
def iterClustering(subset, parsedData, iterNum, iterCount, K):
    # Creating local object
    km = MyKmeans()
    cluster_K = []  # Initialize an empty array
    for x in range(iterNum):
        temp = km.cluster(subset, iterCount, K, [])
        cluster_K.append(temp)
    cluster_K = np.array(cluster_K)
    SC_avg_k = calAvgSC(cluster_K, parsedData)
    return SC_avg_k

# Function to plot the results
def plotAvgSC(K_vals, avgSC_vals):
    plt.figure()
    axes = plt.gca()
    plt.plot(K_vals, avgSC_vals, '-bo')
    plt.title('Average Silhouette Coefficient vs. K - By: Tomoki Koike')
    plt.xlabel('K')
    plt.ylabel('Silhouette Coefficient')
    axes.set_ylim([-1.0, 1.0])
    plt.grid(b=True, which='major', color='#666666', linestyle='--')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig('avgSC_vs_K_plot1_1.png')

# Executing the main function
if __name__ == '__main__':
    main()
