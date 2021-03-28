---
weight: 4
title: "Lorentz Attractor"
date: 2021-03-27T21:57:40+08:00
lastmod: 2021-03-28T16:45:40+08:00
draft: false
author: "Kamal SELVAM"
authorLink: "https://kamalselvam.com"
description: "This article shows the simulation of lorentz attractor using python."
resources:
- name: "featured-image"
  src: "featured-image.png"

tags: ["System Dyanmics", "Python"]
categories: ["System Dynamics"]

lightgallery: true
---

## Introduction 

Lorenz System is one of the well studied non-linear model in system dynamics. Even though being well explored and simulated, it is one of the most beautiful simplistic systems that showcase **chaos**. The non-linear differential equations were initially studied by N. Lorenz, a meteorologist during 1963. 

>  $$ \tag{1} \frac {dx}{dt} = \sigma (y - x) $$
>  $$ \tag{2} \frac {dy}{dt} = x(\rho -z ) - y $$ 
>  $$ \tag{3} \frac {dz}{dt} = xy - \beta z $$ 

The equation ```(1), (2) ``` and ```(3)``` are simplified form of heat convection in earth atmosphere. These equations could help to model natural heat convection in fluid dynamics. When a layer of fluid between two parallel planes, exposed to hot temperature on the bottom plane and cold on the other, creates a linear heat gradient between the two plates. The hot fluid near the bottom plane is less dense, when compared to cold fluid on top. This difference in the density causes the cold fluid to move down towards the bottom due to gravity and the hot fluid to raise up. A continous cycle of the fluid movement is established, creating a cell like structure in between the parallel plates called convection cells. This pattern of convection cells are also called **Rayleigh Bernad** convection cells, named after Lord Rayleigh, who sucessfully analyzed the structure in 1916.  
&nbsp;


![Lorentrz](images/rayleigh.png "Rayleigh bernad convection cells")

## Equation 

These equations are highly sensitive to initial conditions, starting the simulation with an initial condition that are infinitesimally different from each other could create a completely different output.

## Implementation

## Conclusion 

## Reference