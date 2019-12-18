# AUTHOR:      TOMOKI KOIKE
# DESCRIPTION: THIS FILE IS AIMED TO VISUALIZE ALL THE DIGITS IN THE GIVEN MSNIST DIGIT DATASET
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
    sub_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Creating the subset
    subset = createSubset(parsedData, sub_nums)

    #--- The Tasks ---#
    # (1) Visualizing the images based on the 2D features
    visualizeData(subset, sub_nums)

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
        plt.savefig('visualImage_{0}.png'.format(i))
    return

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

if __name__ == '__main__':
    main()