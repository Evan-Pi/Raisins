# Raisins
An attempt of grouping swarms of 3D data points within spheres, for now ;-).

![Alt text](raisins.jpg?raw=true "Title")


In binary classification for separable groups of 2d data points we tend to find the equation of a line through some clever machine learning algorithms so as to split our 2d space into subspaces and then we classify one group of data points from another according to which subspace these points lie within.

We could also draw a circle around each group that includes all the points of the known group and for new data points we could decide to which group they belong with a simple if statement. If the new data points are within the boundaries of the circle then they belong to this particular group, otherwise they do not.

This is, of course, a very simple approach but these are the first steps of an alternative way of doing binary classification. This code intends to produce a new sophisticated way of doing binary classification in higher dimensions!

This repository includes python code for achieving this in 3D space so instead of circles we will play with spheres and after that with raisins!

Download or clone the repository and run the raisin.py file. Also check the data folder to get an idea about the data we use for this demonstration.
