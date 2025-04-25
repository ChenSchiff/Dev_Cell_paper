# Code for "Coordination between endoderm progression and mouse gastruloid elongation controls endodermal morphotype choice"

This repository contains the Python code used for the analysis presented in the Developmental Cell paper: **"Coordination between endoderm progression and mouse gastruloid elongation controls endodermal morphotype choice"**(https://www.cell.com/developmental-cell/fulltext/S1534-5807(24)00335-6)](https://doi.org/10.1016/j.devcel.2024.05.017).

## Overview

This Python script performs a machine learning analysis using decision trees to predict the definitive endoderm (DE) morphotype in mouse gastruloids. It utilizes expression and morphology measurements from an Excel file to train and evaluate multiple decision tree classifiers. The script explores the key predictive parameters and their relationships in determining DE morphotype.

The main steps involved are:

1.  **Data Loading and Preprocessing:** Loading experimental data from an Excel file and preparing it for machine learning.
2.  **Decision Tree Modeling:** Iteratively training decision tree classifiers with maximum depths of 2 and 3 using train-test splits.
3.  **Model Evaluation:** Assessing the performance of the trained trees based on their accuracy on the test data.
4.  **Tree Visualization:** Saving visualizations of high-performing decision trees to understand the decision-making process.
5.  **Feature Importance Analysis:** Analyzing the frequency of top-level splitting parameters in the trees to identify key predictors.
6.  **Parameter Co-occurrence Analysis:** Generating a heatmap to visualize how often different parameters appear together in the initial levels of the shallower decision trees.
7.  **Performance Visualization:** Creating histograms of the decision tree accuracy scores to understand the distribution of model performance.


The script utilizes libraries such as `pandas` for data manipulation, `scikit-learn` for decision tree modeling and evaluation, `graphviz` for tree visualization, and `matplotlib` and `seaborn` for plotting.
