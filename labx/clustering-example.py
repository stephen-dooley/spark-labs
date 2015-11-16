
# coding: utf-8

# #![Spark Logo](http://spark-mooc.github.io/web-assets/images/ta_Spark-logo-small.png) + ![Python Logo](http://spark-mooc.github.io/web-assets/images/python-logo-master-v3-TM-flattened_small.png)
# # **Clustering Examples**
# #### Test the Apache Spark Clustering techniques.
# #### ** This notebook covers: **
# #### *Part 1:* Imports
# #### *Part 2:* Clustering
# 
# #### ** The implementation in MLlib has the following parameters: **
# #### - k is the number of desired clusters.
# #### - maxIterations is the maximum number of iterations to run.
# #### - initializationMode specifies either random initialization or initialization via k-means||.
# #### runs is the number of times to run the k-means algorithm (k-means is not guaranteed to find a globally optimal solution, and 
# #### when run multiple times on a given dataset, the algorithm returns the best clustering result).
# #### - initializationSteps determines the number of steps in the k-means|| algorithm.
# #### - epsilon determines the distance threshold within which we consider k-means to have converged.

# ### ** Part 1: Imports **

# In[1]:

from pyspark.mllib.clustering import KMeans
from numpy import array
from math import sqrt

print "Imports Good"


# #### ** (2) Clustering **

# In[3]:

# Load and parse the data
data = sc.textFile("data/mllib/kmeans_data.txt")
parsedData = data.map(lambda line: array([float(x) for x in line.split(' ')]))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 2, maxIterations=10,
        runs=10, initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


# In[ ]:



