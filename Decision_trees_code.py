import os
from pathlib import Path
import pandas as pd
from gastruloids_functions.functions import *
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.metrics import classification_report


folder_path = 'W:/Experiments/Naama/mp1_gastruloids/Analysis/'  # Folder that contains the excel file
file_name = 'WT_experiments.xlsx'  # The excel file name
sheet_name = 'Sheet4'  # The sheet that contains the data
indexes = 'positions'  # The positions header
rows_number = 60  # The number of measurements rows

if os.path.exists(folder_path):
    print('found path')
else:
    print('cant find path')

# Loading the excel data
df = pd.read_excel(folder_path + file_name, sheet_name=sheet_name, index_col=indexes, skiprows=1, nrows=rows_number)
df = df.fillna(0)

# Decision tree data
X = df.iloc[:, 1:]
Y = df.iloc[:, 0]
features = df.columns[1:]
classes = extract_representing_values_from_list(Y.tolist())

param_names_list = list(X.columns)
param_index_list = list(range(len(X.columns)))
# Creating a dictionary with the parameter names and their coordinated column indexes
df_param_dic = {param_index_list[i]: param_names_list[i] for i in range(len(param_index_list))}


# ---------------------------------------------------------------------------------------------------------------
# Heat map - presents the matches number between parameters from first two rows

def pairs_color_map(df_pairs, save_path=None):
    plt.clf()
    plt.figure(figsize=(14,22))
    sns.heatmap(df_pairs, cmap='coolwarm', annot=False)
    plt.title('Nodes parameters count')
    plt.xlabel('Row No.1 Parameters', fontsize=17)
    plt.ylabel('Row No.2 Parameters (L,R)', fontsize=17)
    plt.xticks(fontsize=16, rotation=90)
    plt.yticks(fontsize=16, rotation=0)
    # plt.yticks(fontsize=3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()

def printf(format, *values):
    a = print(format % values )
    return(str(a))


def extracting_feature_from_tree(features_array, dic):
    list_row1 = []  # empty list for index value in the node at the top
    list_left_row2 = []  # empty list for index value in the node at second row, left side
    list_right_row3 = []  # empty list for index value in the node at second row, right side
    flag2 = 0  # Checking if there is a right node in the second row
    list_row1.append(features_array[0])
    for i in range(1, len(features_array), 1):
        if features_array[i] >= 0:
            if flag2 == 0:
                list_left_row2.append(features_array[i])
                flag2 = 1
            else:
                list_right_row3.append(features_array[i])
    # replacing the index with the parameter name using dictionary
    replaced_list1 = [x if x not in dic else dic[x] for x in list_row1]
    replaced_list2 = [x if x not in dic else dic[x] for x in list_left_row2]
    replaced_list3 = [x if x not in dic else dic[x] for x in list_right_row3]
    return replaced_list1, replaced_list2, replaced_list3


top_parameter_number = []  # Storing the top parameter index
accuracy_score_list = []  # Storing the accuracy score of each tree

concat_data = pd.DataFrame()  # empty df to store the parameter node names
temp_data = pd.DataFrame()  # empty df, temp for concat_data
flag = 0  # Checking if the first row in concat_data was stored

# Creating new folder to save the trees in
Path(folder_path + 'exp12_analysis/EXP12_Tube_no72Sox17int2').mkdir(parents=True, exist_ok=True)

# Creating a range of trees
for i in range(500):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=i)
    # x_train=X
    # y_train=Y
    # x_test=X
    # y_test=Y
    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf = clf.fit(x_train, y_train)

    # Extracting the accuracy value of each tree
    train_score = clf.score(x_train, y_train)
    accuracy_score = clf.score(x_test, y_test)
    if accuracy_score >= 0.6:
        print("train = " + str(train_score) + " test = " + str(accuracy_score) + "\n")
        top_para_name = df_param_dic[clf.tree_.feature[0]] # Extracting the top parameter name
        # Saving the tree plot
        tree_plot = tree.export_graphviz(clf, out_file=None, feature_names=features, class_names=classes,
                                            filled=True, rounded=True, special_characters=True)
        graph_cls = graphviz.Source(tree_plot)
        graph_cls.render(folder_path + 'exp12_analysis/EXP12_ball_no72Sox17int/' + top_para_name + '_' + str(accuracy_score) + '_' + str(i))  # The file name
        # Accuracy test
        top_parameter_number.append(clf.tree_.feature[0])
        # Building the df
        row1_list, row2_left_list, row2_right_list = extracting_feature_from_tree(clf.tree_.feature, df_param_dic)
        temp_data['Row1'] = row1_list
        temp_data['Row2_left'] = row2_left_list
        if row2_right_list:  # Checking if the list is empty
            temp_data['Row2_right'] = row2_right_list
        if flag == 0:
            concat_data_by_rows = pd.concat([concat_data, temp_data], axis=0)
            flag = 1  # The first row was stored
        else:
            concat_data_by_rows = pd.concat([concat_data_by_rows, temp_data], axis=0)
    accuracy_score_list.append(accuracy_score)

# Histogram of accuracy scores
plt.hist(accuracy_score_list, range=[0, 1])
plt.title('Decision tree score histogram')
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

#-------------------------------------------------------------------------------------------------------------------#
# Trees figures and accuracy scores histogram only, without heatmap
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
            folder_path + '3levels_DT/EXP12_ball_no72Sox17int/' + top_para_name + '_' + str(
                accuracy_score) + '_' + str(i))  # The file name
    # Accuracy test
    accuracy_score_list.append(accuracy_score)

plt.hist(accuracy_score_list, range=[0, 1])
plt.title('Decision tree score histogram')
plt.xlabel('DT accuracy score')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig(folder_path + '3levels_DT/EXP12_ball_no72Sox17int/accuracy_histogram', dpi=300)
plt.close()
#------------------------------------------------------------------------------------------------------------------#

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
plt.title('')
plt.ylabel('Parameter count')
plt.xlabel('Frequency as top parameter')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()



