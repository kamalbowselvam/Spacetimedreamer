---
weight: 4
title: "Perceptron"
date: 2020-06-22T21:57:40+08:00
lastmod: 2020-01-01T16:45:40+08:00
draft: false
author: "Kamal SELVAM"
authorLink: "https://kamalselvam.com"
description: "implementation of Preceptron"
resources:
- name: "featured-image"
  src: "featured-image.png"

tags: ["Machine Learning", "Python"]
categories: ["Machine Learning"]

lightgallery: true
---


## Introduction

The single-layer perceptron is one of the simplest neural network architectures. It's a linear classifier used for binary classification tasks. In this article, we'll break down the single-layer perceptron algorithm and implement it in Python. The single-layer perceptron consists of only one layer of artificial neurons, which is also called the output layer. It takes a set of input features, applies weights to them, sums up the weighted inputs, and then applies an activation function to produce an output.

## Implementation 

Intially a python class that accomadate the fuctions to intialize the hyper parameter to run the preceptron like the learning rate, number of iterations and the plot object for visualization purpose. 

```python
from Visualization import DecisionBoundary
import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):


    def __init__(self,learning_rate=0.01, n_iter=100, random_state=42):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], []
        self.ln, = plt.plot([], [], 'ro')


    def fit(self,X,y):
        self.db = DecisionBoundary(X,y)
        random_generator = np.random.RandomState(self.random_state)
        self.weights = random_generator.normal(loc=0.0, scale=0.01, size= X.shape[1])
        self.bias = random_generator.randint(1,1000)
        self.errors_ = []


        for i in range(self.n_iter):
            self.iteration = i

            errors = 0
            for xi, target in zip(X,y):
                error = target - self.predict(xi)
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += int(self.learning_rate * error != 0.0)

            self.db.plot_decision_boundary(self,self.iteration)
            self.errors_.append(errors)

        return self

    def activation_function(self,X):
        return np.dot(X,self.weights) + self.bias


    def predict(self, X):
        return np.where(self.activation_function(X) >= 0.0, 1, -1)
```

The ```fit(self,X,y)``` method takes two parameters ```X``` and ```y``` , where ```X``` is the input feature vector of the data and ```y``` is the target vector or the output vector respectively. ```activation_fuction(X)``` method underhood implements a linear model as defined below:

> $$   y = W \cdot X + b $$

In the above equation W and b are the weight and bias respectively, which are intialized randomly. A unit step is used as a activation function for the linear model. The activation fuction makes sure the value of the output is scaled between -1 and 1. Gradient Descent method is used to find the optimal parameter of the model by iterating over the data.