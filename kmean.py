### FINAL PROJECT ###
## CLASS KMEAN.PY
# AUTHOR:      TOMOKI KOIKE
# DESCRIPTION: THIS FILE INCLUDES THE CLASS TO CREATE AN OBJECT OF MYKMEAN
###

# The modules necessary
import random
import pandas as pd
import numpy as np

class MyKmeans:
    # Function to read the data file
    def readData(self, datafile):
        # Creating the pandas dataframe by importing the data file
        data = pd.read_csv(datafile, names=['ID', 'Digit Label', 'x', 'y'])
        return data

    # Function to perform clustering
    def cluster(self, parsedData, iterCount, k, centroids):
        # Creating a local object to use another function in the class
        araAra = MyKmeans()
        # There are two conditions for the centroids: a preset centroid by user or
        # a randomly generated centroid if the passed argument centroid is empty
        if not centroids:
            centroid_new = araAra.randCentroid(parsedData, k, centroids)
        else:
            centroid_new = araAra.setCentroid(parsedData, k, centroids)
        # Grouping/clustering the data into a new matrix indicating which data is closest to which
        # centroid
        cluster_label = araAra.createClusterLabel(k)
        counterpart = pd.DataFrame(columns=cluster_label)  # Initialize empty dataframe
        for n in range(iterCount):
            # Calculating the square Euclidean distance
            sqEucDist_df = araAra.sqEucDist(parsedData, k, centroid_new)
            group_df = araAra.grouping(sqEucDist_df, k)
            if np.array_equal(np.array(group_df), np.array(counterpart)):
                counterpart = group_df
                # Creating new centroid from the clustered result
                centroid_new = araAra.newCentroidByAverage(parsedData, group_df, k)
            else:
                break
        # Returning the final output
        return araAra.createClusterOutput(parsedData, group_df, k)

    # Function to create silhouette coefficient
    def calculateSC(self, clusters, parsedData):
        # Generate a local object
        jojo = MyKmeans()
        # Obtain the length of each cluster
        clustersLen = list(map(len, clusters))
        # Initialize the silhouette coefficient
        SC = 0
        # Loop through each cluster
        for idx, x in enumerate(clusters):
            # Computation for A (same cluster)
            # Obtain necessary matrices for calculations
            xvals = np.array(parsedData.iloc[x, 2])
            x_temp = xvals[:, np.newaxis]
            xvals = np.repeat(x_temp, clustersLen[idx], axis=1)
            yvals = np.array(parsedData.iloc[x, 3])
            y_temp = yvals[:, np.newaxis]
            yvals = np.repeat(y_temp, clustersLen[idx], axis=1)
            # Transposed and multiplied by (-1)
            xvals_T = np.transpose(x_temp)*(-1)
            yvals_T = np.transpose(y_temp)*(-1)
            # Conduct matrix multiplication
            Xs = xvals + xvals_T
            Ys = yvals + yvals_T
            # Sum the x and y after squaring each term
            XYs = np.square(Xs) + np.square(Ys)
            # Square root all the elements (*these are the distance values)
            XYs_dist = np.sqrt(XYs)
            # Calculating the average
            if not ((clustersLen[idx] == 0) | (clustersLen[idx] == 1)):
                XY_avg = np.sum(XYs_dist, axis=1) / (abs(clustersLen[idx]-1))
                XY_avg = XY_avg.reshape(XY_avg.shape[0], 1)
            else:
                XY_avg = np.zeros((XYs_dist.shape[0], 1))

            # Computation for B (with other clusters)
            temp_clusters = np.array(clusters)
            temp_clusters = np.delete(temp_clusters, idx, axis=0)
            temp_clustersLen = np.array(clustersLen)
            temp_clustersLen = np.delete(temp_clustersLen, idx)
            XY_min = (jojo.SC_DiffClusterAvg(x_temp, y_temp, parsedData,
                                                          temp_clusters, temp_clustersLen))
            # Concatenate the average series and minimum series
            XY_avgMin_comb = np.concatenate((XY_avg, XY_min), axis=1)
            # Calculating the value S that makes up the coefficient incrementally
            S = (XY_min - XY_avg) / np.amax(XY_avgMin_comb, axis=1).reshape(clustersLen[idx], 1)
            SC += sum(S)
        return SC / sum(clustersLen)

    # Function to calculate the average distance from point xi in one cluster to other clusters
    def SC_DiffClusterAvg(self, x_temp, y_temp, parsedData, temp_clusters, temp_clustersLen):
        for idx, x in enumerate(temp_clusters):
            targetX = np.repeat(x_temp, temp_clustersLen[idx], axis=1)
            targetY = np.repeat(y_temp, temp_clustersLen[idx], axis=1)
            xvals = np.array(parsedData.iloc[x, 2])
            xvals = xvals[:, np.newaxis]
            yvals = np.array(parsedData.iloc[x, 3])
            yvals = yvals[:, np.newaxis]
            # Transposed and multiplied by (-1)
            xvals_T = np.transpose(xvals)*(-1)
            yvals_T = np.transpose(yvals)*(-1)
            # Conduct matrix multiplication
            Xs = targetX + xvals_T
            Ys = targetY + yvals_T
            # Sum the x and y after squaring each term
            XYs = np.square(Xs) + np.square(Ys)
            # Square root all the elements (*these are the distance values)
            XYs_dist = np.sqrt(XYs)
            # Calculating the average
            if temp_clustersLen[idx] != 0:
                XYs_avg = np.sum(XYs_dist, axis=1) / (temp_clustersLen[idx])
                XYs_avg = XYs_avg.reshape(XYs_avg.shape[0], 1)
            else:
                XYs_avg = np.zeros((XYs_dist.shape[0], 1))
            if idx == 0:
                avgDist = XYs_avg
            else:
                avgDist = np.concatenate((avgDist, XYs_avg), axis=1)
        if len(temp_clusters) == 1:
            XY_min = avgDist
        else:
            XY_min = np.amin(avgDist, axis=1)
            XY_min = XY_min.reshape(XY_min.shape[0], 1)
        return XY_min

    # Function to create a list<list<int>> output for the function cluster
    def createClusterOutput(self, parsedData, group_df, k):
        # Create local object
        uhauha = MyKmeans()
        # Preallocating an empty list to create output
        alist = []
        # Loop to go through the clustered matrix group_df and find the corresponding IDs
        # in the parsed data
        for x in range(k):
            label_list = uhauha.createClusterLabel(k)
            # Find the indices where the value is 1
            idx = group_df.index[group_df[label_list[x]] == 1].tolist()
            # Append the corresponding IDs to the preset list
            temp = parsedData.iloc[idx]
            temp = np.array(temp['ID'])
            alist.append(temp)
        return alist

    # Function to create labels for the cluster
    def createClusterLabel(self, k):
        label_list = []
        for i in range(k):
            label_list.append('k={}'.format(i+1))
        return label_list

    # Function to create the next centroid by taking the averages of each clusters
    def newCentroidByAverage(self, parsedData, group_df, k):
        # Preallocating a matrix to store all the randomly generated centroid values
        centroids = np.zeros([k, 2])
        for x in range(k):
            astring = 'k={}'.format(x+1)
            # Finding indices where the each data point is in what group (k=?)
            tempIdx = group_df.index[group_df[astring] == 1].tolist()
            # Calculating the mean depending on the indices found above and assigning as new
            # centroid (x, y)
            centroids[x, 0] = parsedData.iloc[tempIdx, 2].mean()
            centroids[x, 1] = parsedData.iloc[tempIdx, 3].mean()
        return centroids

    # Function to assign each data to a group depending on there minimum distance
    def grouping(self, sqEucDist_df, k):
        # Find the minimum in the row and assign a 1 to the column which indicates the group
        # number and 0s to the other columns
        # First we have to preallocate a numpy matrix
        group_mat = np.zeros([(list(sqEucDist_df.shape)[0]), k])
        # Create a dataframe from the numpy matrix
        group_df = pd.DataFrame(group_mat, columns=sqEucDist_df.columns)
        # Find the minimum index in each rows of sqEucDist_df
        minIdx = sqEucDist_df.idxmin(1)
        # Loop to fill in the zero matrix accordingly to sqEucDist_df
        for x in range(k):
            tempIdx = minIdx.index[minIdx == group_df.columns[x]].tolist()
            group_df.iloc[tempIdx, x] = 1
        return group_df

    # Function to calulate the square Euclidean distance for each data point
    def sqEucDist(self, parsedData, k, centroids):
        # Preallocate a matrix to store all the distance values
        size = list(parsedData.shape)[0]  # Number of rows in the dataframe
        sqEucDist_mat = np.zeros([size, k])
        columnLabel = []  # Initialize a column label list
        for n in range(k):
            # Loop to calculate the square Euclidean distance
            sqEucDist_mat[:, n] = (np.sqrt(np.square(parsedData['x'] - centroids[n, 0]) +
                                           np.square(parsedData['y'] - centroids[n, 1])))
            label = 'k={0}'.format(n+1)
            columnLabel.append(label)
        # Change the numpy matrix into a pandas dataframe
        sqEucDist_df = pd.DataFrame(sqEucDist_mat, columns=columnLabel)
        return sqEucDist_df

    # Function to determining the range of the x values and y values
    def findRange(self, parsedData):
        xmax = parsedData['x'].max()  # The maximum value of x data
        xmin = parsedData['x'].min()  # The minimum value of x data
        ymax = parsedData['y'].max()  # The maximum value of y data
        ymin = parsedData['y'].min()  # The minimum value of y data
        return xmax, xmin, ymax, ymin

    # Function to generate random centroids
    def randCentroid(self, parsedData, k, centroid):
        # Creating a local object in this function to use a function in the class
        mymy = MyKmeans()
        # Retrieving the ranges for the x values and y values from the data
        xmax, xmin, ymax, ymin = mymy.findRange(parsedData)
        # Preallocating a matrix to store all the randomly generated centroid values
        new_centroid = np.zeros([k, 2])
        # Creating k random centroids
        for x in range(k):
            random.seed(random.randint(0, 1111))
            # Creating a random x value in the range of x values of the data
            tempX = (xmax - xmin)*random.random() + xmin
            # Assign it to the preallocated np matrix
            new_centroid[x, 0] = tempX
            # Creating a random y value in the range of y values of the data
            tempY = (ymax - ymin)*random.random() + ymin
            # Assign it to the preallocated np matrix
            new_centroid[x, 1] = tempY
        return new_centroid

    # Function to make the centroid array according to the input IDs in the argument 'centroids'
    def setCentroid(self, parsedData, k, centroid):
        # Creating a local object in this function to use another function in the class
        nani = MyKmeans()
        # Preallocating a matrix to store all the randomly generated centroid values
        new_centroid = np.zeros([k, 2])
        for x in range(k):
            # Finding the index that has the matching ID# to the input centroid
            indexX = parsedData.index[parsedData['ID'] == centroid[x]].tolist()
            # Assigning the centroid x-value to a variable
            tempX = parsedData.iloc[indexX[0], 3]
            # Inputting the x centroid value to the matrix
            new_centroid[x, 0] = tempX
            indexY = parsedData.index[parsedData['ID'] == centroid[x]].tolist()
            tempY = parsedData.iloc[indexY[0], 4]
            new_centroid[x, 1] = tempY
        return new_centroid