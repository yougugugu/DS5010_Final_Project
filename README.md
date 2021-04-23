# DS5010 Final Project

![GitHub top language](https://img.shields.io/github/languages/top/Linzzz81/DS5010_Final_Project.svg)

This project is designed as the final project of Northeastern University DS5010

## TeamMember

Team member:

| Name        | NuID      |
| ----------- | --------- |
| [@Lin Zhu](https://github.com/Linzzz81)     | 001066973 |
| [@Yihao Gu](https://github.com/yougugugu)   | 001305641 |

## Introduction

Our goal is to design a linear regression package, which implements most parts of a mutiple linear regression based on ordinary least square. 
This package provides caculation of each parameter and coefficient of linear regression in ```lr.py``` module including **t-test**, **f-test**, **Rsquared** and **SST, SSE, SSR**. 
It also contains functions in ```da.py``` module to help process data, like **read_data()** to read data from local, **partition()** to split dataset into training and validationset, **select_byindex(), select_byname()** to slice dataset by columns. 
Additionally, ```diagnose.py``` module provides diagnostic tools for linear regreesion including **leverage()**, **cooks_distance()** and **plot()** to help diagnose linear regression based on different plots.


## Running the tests

Open terminals and ```cd``` to ```...\DS5010_Final_Project``` dictionary. Type ```py run_tests.py``` to run all unit tests.

## Built With

* [Python](https://www.python.org/) - The programming language to implement the program.

