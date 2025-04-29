# -*- coding: utf-8 -*-
"""Decision_trees_code.ipynb
"""

# ------------------------------------------------------------------------------
# Script Name: Decision_trees_code.py
# Author: Chen Sacharen
# Date Created:
# Last Modified: 2024-04-25
#
# Description:
# This script performs a machine learning analysis using decision trees to predict
# the definitive endoderm (DE) morphotype in mouse gastruloids. It utilizes
# expression and morphology measurements from an Excel file
# to train and evaluate multiple decision tree classifiers.
#
# Key Processes:
# 1. Loads data from the specified Excel sheet.
# 2. Preprocesses the data by handling missing values.
# 3. Iteratively trains decision tree models (with max depths of 2 and 3) using
#    train-test splits.
# 4. Evaluates the accuracy of each trained tree.
# 5. Saves visualizations of high-performing decision trees.
# 6. Analyzes the frequency of top-level parameters in the trees.
# 7. Generates a heatmap visualizing the co-occurrence of parameters in the
#    initial levels of the trees (for max depth 2).
# 8. Creates histograms of the accuracy scores for both tree depths.
#
# Input Data:
# - Excel file: containing gastruloid measurements, with 'positions' as the
#   index column and the first data column representing the DE morphotype.
#
# Output:
# - Saved PNG images of decision tree visualizations in specified folders.
# - Histograms of decision tree accuracy scores.
# - A heatmap visualizing parameter co-occurrence.
# - A bar plot showing the frequency of top-level parameters.
#
# Related Publication:
# - Developmental Cell paper: "Coordination between endoderm progression and
#   mouse gastruloid elongation controls endodermal morphotype choice."
# ------------------------------------------------------------------------------

# --- Import libraries ---
import os
from pathlib import Path
import pandas as pd
from gastruloids_functions.functions import * # Assuming this file contains custom functions relevant to gastruloid analysis
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
folder_path = 'W:/Experiments/Naama/mp1_gastruloids/Analysis/'  # Folder containing the input Excel file
file_name = 'WT_experiments.xlsx'  # Name of the Excel file
sheet_name = 'Sheet4'  # Name of the sheet within the Excel file containing the data
indexes = 'positions'  # Name of the column to be used as the index
rows_number = 60  # Number of data rows to read from the sheet

# Check if the specified folder path exists
if os.path.exists(folder_path):
    print('found path')
else:
    print('cant find path')

# --- Data Loading and Preparation ---
# This section handles loading the experimental data from the Excel file,
# setting the index, and separating the features (X) from the target variable (Y).

# Load the data from the Excel file into a pandas DataFrame
df = pd.read_excel(folder_path + file_name, sheet_name=sheet_name, index_col=indexes, skiprows=1, nrows=rows_number)
# Fill any missing values with 0
df = df.fillna(0)

# Separate features (X) and target variable (Y) for the decision tree data
Y = df.iloc[:, 0] # The first column is assumed to be the target variable (DE morphotype)
X = df.iloc[:, 1:] # All subsequent columns are features (expression and morphology measurements)
features = df.columns[1:] # List of feature names
classes = extract_representing_values_from_list(Y.tolist()) # Extract unique morphotype labels

# Create a dictionary mapping feature indices to their names
param_names_list = list(X.columns)
param_index_list = list(range(len(X.columns)))
df_param_dic = {param_index_list[i]: param_names_list[i] for i in range(len(param_index_list))}

# ---------------------------------------------------------------------------------------------------------------
# Heatmap Analysis of Decision Tree Node Parameters Functions
# ---------------------------------------------------------------------------------------------------------------
# This section defines functions and implements the analysis to visualize the
# co-occurrence of parameters in the initial splits (top two levels) of the
# decision trees generated with a maximum depth of 2. The goal is to identify
# which parameters frequently appear together at the top of the predictive models.
#
# Key Functions:
# - pairs_color_map(df_pairs, save_path=None): Generates and saves a heatmap
#   showing the count of times each pair of parameters (from the top and
#   second levels of the trees) appear together.
# - extracting_feature_from_tree(features_array, dic): Extracts the feature
#   indices from the first two levels of a given decision tree and maps them
#   back to their original parameter names using the provided dictionary.
#
# Analysis Workflow:
# 1. The 'extracting_feature_from_tree' function is used within the first
#    decision tree training loop (max_depth=2) to get the parameter names
#    at the root and the first two child nodes for each tree.
# 2. These parameter names are stored and organized into a DataFrame.
# 3. The 'two_columns_frequencies_df' function (is in
#    'gastruloids_functions.functions') calculates the frequency of each
#    parameter pair between the top node ('Row1') and the combined left
#    and right child nodes ('Row2').
# 4. The 'pairs_color_map' function then visualizes these frequencies as a
#    heatmap, allowing for the identification of potentially important
#    parameter relationships in predicting DE morphotypes.
#
# Output:
# - A heatmap image jpg saved in the specified analysis folder, showing
#   the co-occurrence counts.
# ---------------------------------------------------------------------------------------------------------------

def pairs_color_map(df_pairs, save_path=None):
    """
    Generates and saves a heatmap visualizing the co-occurrence of parameters
    in the first two levels of the decision trees.

    Args:
        df_pairs (pd.DataFrame): DataFrame containing pairs of parameters.
        save_path (str, optional): Path to save the heatmap image.
    """
    plt.clf()
    plt.figure(figsize=(14,22))
    sns.heatmap(df_pairs, cmap='coolwarm', annot=False)
    plt.title('Nodes parameters count')
    plt.xlabel('Row No.1 Parameters', fontsize=17)
    plt.ylabel('Row No.2 Parameters (L,R)', fontsize=17)
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


def printf(format, *values):
    """
    Custom print function that returns the printed string.
    """
    a = print(format % values )
    return(str(a))


def extracting_feature_from_tree(features_array, dic):
    """
    Extracts the feature indices from the first two levels of a decision tree
    and maps them to their corresponding parameter names using the provided dictionary.

    Args:
        features_array (list): Array of feature indices in the decision tree.
        dic (dict): Dictionary mapping feature indices to parameter names.

    Returns:
        tuple: A tuple containing three lists:
            - replaced_list1: List containing the parameter name of the top node.
            - replaced_list2: List containing the parameter name(s) of the left child node (if exists).
            - replaced_list3: List containing the parameter name(s) of the right child node (if exists).
    """
    list_row1 = []  # Empty list for the index value in the top node
    list_left_row2 = []  # Empty list for the index value in the left child node
    list_right_row3 = []  # Empty list for the index value in the right child node
    flag2 = 0  # Flag to track if the left child node has been processed
    list_row1.append(features_array[0])
    for i in range(1, len(features_array), 1):
        if features_array[i] >= 0:
            if flag2 == 0:
                list_left_row2.append(features_array[i])
                flag2 = 1
            else:
                list_right_row3.append(features_array[i])
    # Replace the index with the parameter name using the dictionary
    replaced_list1 = [x if x not in dic else dic[x] for x in list_row1]
    replaced_list2 = [x if x not in dic else dic[x] for x in list_left_row2]
    replaced_list3 = [x if x not in dic else dic[x] for x in list_right_row3]
    return replaced_list1, replaced_list2, replaced_list3

# ---------------------------------------------------------------------------------------------------------------
# Decision Tree Analysis Variables
# ---------------------------------------------------------------------------------------------------------------
# This section initializes variables that will be used in the subsequent
# decision tree training and analysis process. These variables store information
# extracted from the trained trees, such as the top-level splitting parameter,
# the accuracy of each tree, and the parameter names at the first two levels.
#
# Variables:
# - top_parameter_number (list): Stores the index of the feature used for the
#   root split of each trained decision tree that meets the accuracy threshold.
# - accuracy_score_list (list): Stores the testing accuracy score for each
#   trained decision tree that meets the accuracy threshold.
# - concat_data (pd.DataFrame): An empty DataFrame that will be populated with
#   the parameter names from the top two levels of the qualified decision trees.
# - temp_data (pd.DataFrame): A temporary DataFrame used to hold the parameter
#   names of a single tree before being concatenated to 'concat_data'.
# - flag (int): A flag variable used to track whether the 'concat_data' DataFrame
#   is currently empty (flag == 0) or has been populated with data (flag == 1).
#
# Output:
# - Creates a directory (if it doesn't exist) to store the generated decision
#   tree plot images.
# ---------------------------------------------------------------------------------------------------------------

top_parameter_number = []  # List to store the index of the top parameter of each tree
accuracy_score_list = []  # List to store the accuracy score of each trained tree
concat_data = pd.DataFrame()  # Empty DataFrame to store the parameter names from the top two levels of the trees
temp_data = pd.DataFrame()  # Temporary DataFrame to hold parameter names for each tree
flag = 0  # Flag to check if the first row in concat_data was stored

# Create a new folder to save the generated decision tree plots
Path(folder_path + 'exp12_analysis/EXP12_Tube_no72Sox17int').mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------------------------------------------
# Decision Tree Training and Analysis (Max Depth 2)
# ---------------------------------------------------------------------------------------------------------------
# This section implements an iterative process of training and evaluating
# decision tree classifiers. It performs 500 iterations, each involving:
# 1. Splitting the dataset into training (70%) and testing (30%) sets using a
#    unique random state to ensure variability across iterations.
# 2. Initializing a decision tree classifier with a maximum depth of 2.
# 3. Training the classifier on the training data.
# 4. Evaluating the accuracy of the trained model on both the training and
#    testing datasets.
# 5. For trees that achieve a testing accuracy of 0.6 or higher:
#    - Printing the training and testing accuracy scores.
#    - Extracting the name of the parameter used for the top-level split.
#    - Generating and saving a visualization of the decision tree as a PNG file,
#      including the top parameter name and the accuracy in the filename.
#    - Storing the index of the top-level parameter in the 'top_parameter_number' list.
#    - Extracting the parameter names from the root node and the immediate
#      left and right child nodes (first two levels of the tree).
#    - Storing these parameter names in the 'concat_data_by_rows' DataFrame
#      for subsequent heatmap analysis.
#    - Appending the testing accuracy score to the 'accuracy_score_list'.
#
# Input:
# - Features (X) and target variable (Y) derived from the loaded Excel data.
# - The 'df_param_dic' mapping feature indices to parameter names.
#
# Output:
# - Prints training and testing accuracy for qualified trees.
# - Saves PNG images of the visualized decision trees in a specified folder.
# - Populates the 'top_parameter_number' and 'accuracy_score_list'.
# - Populates the 'concat_data_by_rows' DataFrame with parameter names from
#   the top two levels of the qualified trees.
# ---------------------------------------------------------------------------------------------------------------

for i in range(500):
    # Split the data into training and testing sets with a 70/30 ratio and a unique random state for each iteration
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)
    # Initialize a Decision Tree Classifier with a maximum depth of 2
    clf = tree.DecisionTreeClassifier(max_depth=2)
    # Train the decision tree on the training data
    clf = clf.fit(x_train, y_train)

    # Evaluate the accuracy of the trained tree on both training and testing sets
    train_score = clf.score(x_train, y_train)
    accuracy_score = clf.score(x_test, y_test)

    # Process trees with a testing accuracy of 0.6 or higher
    if accuracy_score >= 0.6:
        print("train = " + str(train_score) + " test = " + str(accuracy_score) + "\n")
        # Extract the name of the top-level parameter (feature) used for the first split
        top_para_name = df_param_dic[clf.tree_.feature[0]]
        # Generate a visualization of the decision tree
        tree_plot = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names=classes,
                                        filled=True, rounded=True, special_characters=True)
        graph_cls = graphviz.Source(tree_plot)
        # Save the tree visualization as a PNG file
        graph_cls.render(folder_path + 'exp12_analysis/EXP12_ball_no72Sox17int/' + top_para_name + '_' + f'{accuracy_score:.2f}' + '_' + str(i))
        # Store the index of the top parameter
        top_parameter_number.append(clf.tree_.feature[0])
        # Extract the parameter names from the first two levels of the tree
        row1_list, row2_left_list, row2_right_list = extracting_feature_from_tree(clf.tree_.feature, df_param_dic)
        # Create a temporary DataFrame to store these parameter names
        temp_data['Row1'] = row1_list
        temp_data['Row2_left'] = row2_left_list
        if row2_right_list:  # Check if a right child node exists
            temp_data['Row2_right'] = row2_right_list
        # Concatenate the temporary DataFrame to the main DataFrame
        if flag == 0:
            concat_data_by_rows = pd.concat([concat_data, temp_data], axis=0)
            flag = 1  # Mark that the first row has been added
        else:
            concat_data_by_rows = pd.concat([concat_data_by_rows, temp_data], axis=0)
        # Store the accuracy score
        accuracy_score_list.append(accuracy_score)

# ---------------------------------------------------------------------------------------------------------------
# Visualization of Accuracy Scores and Preparation for Heatmap
# ---------------------------------------------------------------------------------------------------------------
# This section first generates a histogram to visualize the distribution of
# the testing accuracy scores obtained from the decision trees trained in the
# previous section (with a maximum depth of 2). This provides an overview of
# the model performance across the iterations.
#
# Following the histogram generation, this section prepares the data for
# creating a heatmap. It takes the 'concat_data_by_rows' DataFrame, which
# contains the parameter names from the top two levels of the qualified
# decision trees, and restructures it to calculate the co-occurrence frequencies
# of these parameters.
#
# Steps:
# 1. Generates a histogram of the 'accuracy_score_list' to show the distribution
#    of testing accuracies. The histogram is saved as a PNG file.
# 2. Resets the index of the 'concat_data_by_rows' DataFrame.
# 3. Fills any missing values in the DataFrame with empty strings.
# 4. Creates a new DataFrame 'df_node_pairs' to organize the parameter pairs
#    from the top level ('Row1') and the combined second level ('Row2_left'
#    and 'Row2_right'). If a 'Row2_right' column exists, the left and right
#    parameters are concatenated with a comma.
# 5. Resets the index of the 'df_node_pairs' DataFrame.
#
# Output:
# - A histogram image saved in a specified folder, showing the distribution
#   of decision tree accuracy scores.
# - The 'df_node_pairs' DataFrame, which is prepared for the subsequent
#   calculation and visualization of parameter co-occurrence in the heatmap.
# ---------------------------------------------------------------------------------------------------------------

# Histogram of accuracy scores
plt.hist(accuracy_score_list, range=[0, 1])
plt.title('Decision tree score histogram (max_depth=2)')
plt.xlabel('DT accuracy score', fontsize=15)
plt.ylabel('Counts',fontsize=15)
plt.tight_layout()
plt.savefig(folder_path + 'exp12_analysis/EXP12_ball_no72Sox17int/12012023_accuracy_histogram_EXP12_tube', dpi=300)
plt.close()

# Data for heatmap
concat_data_by_rows = concat_data_by_rows.reset_index(drop=True)
concat_data_by_rows = concat_data_by_rows.fillna('')
df_node_pairs = pd.DataFrame()
df_node_pairs['Row1'] = concat_data_by_rows['Row1']
if 'Row2_right' in concat_data_by_rows.columns:
    df_node_pairs['Row2'] = concat_data_by_rows['Row2_left'] + ', ' +\
                            concat_data_by_rows['Row2_right']  # Merging both columns into one
else:
    df_node_pairs['Row2'] = concat_data_by_rows['Row2_left']

df_node_pairs = df_node_pairs.reset_index(drop=True)

# Creating a heatmap of the pair counts for two given columns
df_pairs_count = two_columns_frequencies_df(df_node_pairs, column_name1='Row2', column_name2='Row1')
pairs_color_map(df_pairs_count, save_path='W:/Experiments/Naama/mp1_gastruloids/Analysis/exp12_analysis/EXP12_ball_no72Sox17int/'
                                        '18012023_ball_2levels_accuracy06_EXP12_Parameter_nodes_count_colormap_500.jpg')

# ---------------------------------------------------------------------------------------------------------------
# Decision Tree Analysis (Max Depth 3) - Tree Visualizations and Accuracy Histogram
# ---------------------------------------------------------------------------------------------------------------
# This section focuses on training and visualizing decision tree classifiers
# with a maximum depth of 3. It iterates 500 times, similar to the previous
# section, but with a deeper tree structure. The primary outputs are the
# saved visualizations of the trained trees that meet the accuracy threshold
# and a histogram of their testing accuracy scores. This analysis explores the
# impact of increased tree depth on model performance and the identified key
# parameters.
#
# Key Processes:
# 1. Iterates 500 times, each time splitting the data into training and
#    testing sets with a unique random state.
# 2. Initializes and trains a decision tree classifier with a maximum depth of 3.
# 3. Evaluates the training and testing accuracy of the tree.
# 4. If the testing accuracy is 0.6 or higher:
#    - Prints the training and testing accuracy.
#    - Extracts the name of the top-level splitting parameter.
#    - Generates and saves a visualization of the decision tree as a PNG file
#      in the specified folder, including the top
#      parameter name and accuracy in the filename.
# 5. Appends the testing accuracy score to the 'accuracy_score_list'.
# 6. After the loop, generates and saves a histogram of all the collected
#    accuracy scores for the trees with a maximum depth of 3.
#
# Input:
# - Features (X) and target variable (Y) from the loaded Excel data.
# - The 'df_param_dic' for mapping feature indices to names.
#
# Output:
# - PNG images of visualized decision trees (with max depth 3) saved in a
#   specified folder.
# - A histogram image ('accuracy_histogram.png') in the same folder, showing
#   the distribution of accuracy scores for these deeper trees.
# - Prints training and testing accuracy for qualified trees.
# ---------------------------------------------------------------------------------------------------------------
for i in range(500):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(x_train, y_train)

    # Extracting the accuracy value of each tree
    train_score = clf.score(x_train, y_train)
    accuracy_score = clf.score(x_test, y_test)
    if accuracy_score >= 0.6:
        print("train = " + str(train_score) + " test = " + str(accuracy_score) + "\n")
        top_para_name = df_param_dic[clf.tree_.feature[0]]  # Extracting the top parameter name
        # Saving the tree plot
        tree_plot = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names=classes,
                                        filled=True, rounded=True, special_characters=True)
        graph_cls = graphviz.Source(tree_plot)
        # accstr = printf("_%.2f_", accuracy_score)
        graph_cls.render(
            folder_path + '3levels_DT/EXP12_ball_no72Sox17int/' + top_para_name + '_' + f'{accuracy_score:.2f}' + '_' + str(i))  # The file name
    # Accuracy test
    accuracy_score_list.append(accuracy_score)

plt.hist(accuracy_score_list, range=[0, 1])
plt.title('Decision tree score histogram (max_depth=3)')
plt.xlabel('DT accuracy score')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig(folder_path + '3levels_DT/EXP12_ball_no72Sox17int/accuracy_histogram', dpi=300)
plt.close()

# ---------------------------------------------------------------------------------------------------------------
# Bar Plot of Top Parameter Frequency (Max Depth 3)
# ---------------------------------------------------------------------------------------------------------------
# This section aims to identify the most frequently occurring parameter at the
# root node (top split) of the decision trees trained with a maximum depth of 3
# that meet the specified accuracy threshold (>= 0.6). By counting the occurrences
# of each top parameter across the iterations, this analysis highlights the
# features that are most often selected as the primary predictor of DE morphotype
# in the deeper tree models.
#
# Key Processes:
# 1. Iterates 500 times, training a decision tree with a maximum depth of 3 in
#    each iteration.
# 2. For each trained tree that achieves a testing accuracy of 0.6 or higher,
#    the index of the feature used for the top split is appended to the
#    'top_parameter_number' list.
# 3. After the loop, this section counts the occurrences of each unique top
#    parameter index in the 'top_parameter_number' list.
# 4. These counts are then organized into a pandas DataFrame ('df_top_parameter'),
#    where the parameter indices are mapped back to their original names using
#    the 'df_param_dic'.
# 5. Finally, a bar plot is generated to visualize the frequency of each top
#    parameter, with the parameter names on the x-axis and their occurrence
#    count on the y-axis.
#
# Input:
# - Features (X) and target variable (Y) from the loaded Excel data.
# - The 'df_param_dic' for mapping feature indices to names.
#
# Output:
# - A bar plot showing the frequency of each parameter that appeared
#   as the top node in the qualified decision trees (max depth 3).
# ---------------------------------------------------------------------------------------------------------------
# Bar plot - counts the occurrence number of the top node parameter

for i in range(500):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(x_train, y_train)

    # Extracting the accuracy value of each tree
    accuracy_score = clf.score(x_test, y_test)
    if accuracy_score >= 0.6:
        accuracy_score_list.append(accuracy_score)
        top_parameter_number.append(clf.tree_.feature[0])

check_list = []  # Storing unique values
num_value_list = []  # Storing the index values
num_count_list = []  # Storing the counts for each index
for num in top_parameter_number:
    if num in check_list:
        continue
    num_value_list.append(num)
    num_count_list.append(top_parameter_number.count(num))
    check_list.append(num)

data_tuples = list(zip(num_value_list, num_count_list))
df_top_parameter = pd.DataFrame(data_tuples, columns=['Parameter_name', 'Parameter_count'])
df_top_parameter['Parameter_name'] = df_top_parameter['Parameter_name'].apply(lambda x: df_param_dic[x])

# Bar plot for the counts of each top parameter over decision trees iteration
ax = df_top_parameter.plot.bar(x='Parameter_name', y='Parameter_count', rot=60, legend=False, color='#D3D3D3')
plt.title('Top parameter frequency (max_depth=3)')
plt.ylabel('Parameter count')
plt.xlabel('Frequency as top parameter')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()
