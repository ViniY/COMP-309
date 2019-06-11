from typing import TextIO

import numpy
import matplotlib
import csv
import pandas as pd
import os
import glob

def csvfileReading(path):
    allFiles = glob.glob(path + "/*.csv")
    for file in allFiles:
        df = pd.read_csv(file,header=None,usecols=[0,1])
        fitness_of_each_generation =[]
        fitness_of_each_generation = df[0] * 0.9 + df[1] * 0.1
    fit = -1
    max_fit = -10
    min_fit = 10
    avgFit = -1
    AvgCost = -1
    for row in fitness_of_each_generation:
        if (row ==0):
            fitness_of_each_generation
            x = numpy.delete(fitness_of_each_generation,row,axis=0)
        if(fit < x[row])





def readCSVDirectory(directoryPath):
    directory = os.path.join(directoryPath,"path")
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                f = open(file,'r')
                f.close()
                csvfileReading(f)




 if __name__ == '__main__':
    data = readCSVDirectory()