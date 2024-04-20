---
weight: 4
title: "Gradient Descent"
date: 2019-12-11T21:57:40+08:00
lastmod: 2020-01-01T16:45:40+08:00
draft: false
author: "Kamal SELVAM"
authorLink: "https://kamalselvam.com"
description: "This article shows the implementation of Gradient Descent Algorithm"
resources:
- name: "featured-image"
  src: "featured-image.png"

tags: ["Machine Learning", "Python"]
categories: ["Machine Learning"]

lightgallery: true
---


## Introduction
Gradient descent is a fundamental optimization algorithm widely used in machine learning and optimization problems. It is employed to minimize a function by iteratively moving in the direction of the steepest descent as indicated by the negative of the gradient. This article aims to explain the concepts behind gradient descent and its implementation in python.

Before diving into gradient descent, it's crucial to understand the notion of optimization. In optimization, the goal is to find the minimum or maximum of a function. For simplicity, let's focus on minimizing a function, typically denoted as $ \( f(x) \) $, where $ \( x \) $ represents the parameters of the function. The process of finding the minimum of $ \( f(x) \) $ involves iterative steps towards adjusting the parameters $ \( x \) $ until reaching a minimum.

## The Gradient

The gradient of a function, denoted as $\( \nabla f(x) \) $, is a vector that points in the direction of the steepest increase of the function at a particular point. In other words, it indicates the direction in which the function grows fastest. The negative gradient, $ \( -\nabla f(x) \)$, points in the direction of the steepest decrease, which is the direction of the greatest decrease of the function.

## Gradient Descent Algorithm

Gradient descent operates by iteratively updating the parameters $\( x \)$ in the opposite direction of the gradient of the function $\( f(x) \)$ with respect to $ \( x \) $. The update rule for gradient descent can be represented as:

$$ \[ x_{t+1} = x_t - \alpha \nabla f(x_t) \] $$

where:
- $ \( x_t \) $ represents the parameters at iteration $ \( t \) $.
- $ \( \alpha \) $ (alpha) denotes the learning rate, which controls the step size or the rate at which the parameters are updated.

The learning rate is a critical hyperparameter in gradient descent. A too small learning rate may result in slow convergence, while a too large learning rate can cause divergence, where the optimization process fails to converge to a minimum.

## Implementation 

```python 

def randomDataGenerator():
    """

    :param theta0_init: slope
    :param theta1_init: bias
    :return: data x,y
    """

    x = np.arange(start=0, stop=5, step=0.01)
    n_rnd = 500
    theta0_init = np.random.normal(loc=1, scale=0.1, size=n_rnd)
    theta1_init = np.random.normal(loc=5, scale=0.2, size=n_rnd)
    y = theta0_init * x + theta1_init
    return x , y, theta0_init, theta1_init

```

## Conclusion

Gradient descent is a powerful optimization algorithm used to minimize functions iteratively by moving in the direction of the steepest descent. Understanding gradient descent and its variants is essential for practitioners in machine learning and optimization fields, as it underpins many modern optimization techniques and algorithms. With proper tuning of hyperparameters and careful consideration of the problem domain, gradient descent can efficiently solve a wide range of optimization problems.
