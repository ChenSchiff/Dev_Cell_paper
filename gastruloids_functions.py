import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from random import randrange

# EB feature vs time graph
def eb_feature_vs_time(condition, time_points, y_min, y_max, column_name='', title='', x_title='', y_title='',
                       save_path=None):
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=20, color='black')
    plt.xlabel(x_title, fontsize=16)
    plt.ylabel(y_title, fontsize=16, rotation=0, labelpad=30)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for EB_position in range(len(condition)):
        x_val = []
        y_val = []
        for time in time_points:
            time_header = condition[str(time) + column_name]
            y_value = time_header.values[EB_position]
            if y_value != 0:
                y_val.append(time_header.values[EB_position])
                x_val.append(time)

        plt.plot(x_val, y_val, marker='o')
    # plt.ylim([y_min, y_max])
    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# Scatter matrix plot - one df
def scatter_matrix_plot(df_scatter_matrix, save_path=None):
    plt.clf()
    sns.pairplot(df_scatter_matrix)
    # plt.title('Scatter plot')
    # plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


# Heat map - spearman/Pearson correlation
def heat_map_plot_correlation(df_heat, correlation_method='', save_path=None):
    plt.clf()
    corr_mat = df_heat.corr(method=correlation_method)
    sns.heatmap(corr_mat, vmin=0, vmax=1, cmap='coolwarm')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8, rotation=0)
    # plt.title('Spearman method/Pearson method')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


# Color map
def color_map(df_heat, save_path=None):
    plt.clf()
    sns.heatmap(df_heat, cmap='coolwarm', annot=True)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8, rotation=0)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


# Mean calculation of specific feature, for given time points. Returns a list with the means for each time
def feature_mean_cal(conditions_l, time_column, column_name=''):
    main_mean_list = []
    for condition in conditions_l:
        condition = condition.replace(to_replace=0, value=np.nan)
        mean_list = []
        for time in time_column:
            time_head = str(time) + column_name
            mean_list.append(condition[time_head].mean())
        main_mean_list.append(mean_list)
    return main_mean_list


# Standard deviation calculation of specific feature,for given time points. Returns a list with the means for each time
def feature_std_cal(conditions_l, time_column, column_name=''):
    main_std_list = []
    for condition in conditions_l:
        condition = condition.replace(to_replace=0, value=np.nan)
        std_list = []
        for time in time_column:
            time_head = str(time) + column_name
            std_list.append(condition[time_head].std())
        main_std_list.append(std_list)
    return main_std_list


# Max value of list with sub-lists
def max_sub_list(lst):
    max_value = lst[0][0]
    for sub_lst in lst:
        for i in sub_lst:
            if i > max_value:
                max_value = i
    return max_value


# Min value of list with sub-lists
def min_sub_list(lst):
    min_value = lst[0][0]
    for sub_list in lst:
        for i in sub_list:
            if i < min_value:
                min_value = i
    return min_value


# Error bar graph - more than one group
def error_bar_plot_total(mean_list, std_list, time_points, title='', x_title='', y_title='', save_path=None):
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=20, color='black')
    plt.xlabel(x_title, fontsize=12)
    plt.ylabel(y_title, fontsize=12, rotation=0, labelpad=40)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    for i, j in zip(range(len(mean_list)), range(len(std_list))):
        plt.errorbar(time_points, mean_list[i], yerr=std_list[j], label='Condition' + str(i+1))

    plt.legend(loc=2)
    mean_max_value = max_sub_list(mean_list)
    std_max_value = max_sub_list(std_list)
    mean_min_value = min_sub_list(mean_list)
    std_min_value = min_sub_list(std_list)
    # plt.ylim([mean_min_value - std_min_value*2, mean_max_value + std_max_value*1.5])
    plt.ylim(0, mean_max_value + std_max_value*1.5)
    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# Error bar graph - more than ten groups
def error_bar_plot_total_more_than_ten_conditions(mean_list, std_list, time_points, title='', x_title='', y_title='',
                                                  save_path=None):
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=20, color='black')
    plt.xlabel(x_title, fontsize=12)
    plt.ylabel(y_title, fontsize=12, rotation=0, labelpad=40)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    colors = ['Blue', 'Orange', 'Green', 'Red', 'Purple', 'Brown', 'Pink', 'Gray', 'OliveDrab', 'Cyan', 'Gold', 'Plum']
    conditions_list_name = ['D6D 3uM 24h', 'D5D 10uM 24h', 'DMSO 24h', 'SA 10uM 24h', 'SA 3uM 24h', 'D6D 3uM 48h',
                            'D5D 10uM 48h', 'DMSO 48h']

    for i, j, color, condition in zip(range(len(mean_list)), range(len(std_list)), colors, conditions_list_name):
        new_time_points = adding_int_random_number_to_number_in_a_list(time_points)
        plt.errorbar(new_time_points, mean_list[i], yerr=std_list[j], capsize=10, fmt="o:",
                     label=condition)

    plt.legend(loc=2)
    mean_max_value = max_sub_list(mean_list)
    std_max_value = max_sub_list(std_list)
    mean_min_value = min_sub_list(mean_list)
    std_min_value = min_sub_list(std_list)
    # plt.ylim([mean_min_value - std_min_value*2, mean_max_value + std_max_value*1.5])
    plt.ylim(0, mean_max_value + std_max_value * 1.5)
    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# Error bar graph - one group
def error_bar_plot(mean_list, std_list, time_points, obj_num, title='', x_title='', y_title='', save_path=None):
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(title + str(obj_num + 1), fontsize=20, color='black')
    plt.xlabel(x_title, fontsize=12)
    plt.ylabel(y_title, fontsize=12, rotation=0, labelpad=40)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.errorbar(time_points, mean_list[obj_num], yerr=std_list[obj_num])
    print(std_list[obj_num])

    mean_max_value = max_sub_list(mean_list)
    std_max_value = max_sub_list(std_list)
    mean_min_value = min_sub_list(mean_list)
    std_min_value = min_sub_list(std_list)
    plt.ylim([mean_min_value - std_min_value*2, mean_max_value + std_max_value*1.5])
    plt.ylim(0, 7)
    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# The function receives two df columns and returns a list with the ratio of the two df columns values
def ratio_two_columns(column1, column2):
    ratio_list = []
    list1 = column1.tolist()
    list2 = column2.tolist()
    for value1, value2 in zip(list1, list2):
        ratio_list.append(value1 / value2)
    return ratio_list


# Plot that displays the ratio between two df columns
def one_ratio_data_display_plot(column1_name, column2_name, x_value, title='', x_title='', y_title='', save_path=None):
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=20, color='black')
    plt.xlabel(x_title, fontsize=16)
    plt.ylabel(y_title, fontsize=16, rotation=0, labelpad=30)

    ratio_list = ratio_two_columns(column1_name, column2_name)
    x_value_list = list((len(ratio_list)) * [x_value])
    plt.plot(x_value_list, ratio_list, marker='o')
    plt.ylim([0, 1])
    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# Plot that displaying the ratios between two df columns, for more than one condition
def ratios_data_display_plot(conditions_list, column1_name, column2_name, x_values_list, title='', x_title='',
                             y_title='', save_path=None):
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=20, color='black')
    plt.xlabel(x_title, fontsize=16)
    plt.ylabel(y_title, fontsize=16, rotation=0, labelpad=30)

    for condition, x_value in zip(conditions_list, x_values_list):
        ratio_list = ratio_two_columns(condition[column1_name], condition[column2_name])
        x_value_list = list((len(ratio_list)) * [x_value])
        plt.plot(x_value_list, ratio_list, marker='o')

    plt.ylim([0, 1])
    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# Scatter plot that displays the correlation between two df columns values
def two_columns_data_display(column1_name, column2_name, title='', x_title='', y_title='', save_path=None):
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=20, color='black')
    plt.xlabel(x_title, fontsize=16)
    plt.ylabel(y_title, fontsize=16, rotation=0, labelpad=30)

    column1_list = column1_name.tolist()
    column2_list = column2_name.tolist()
    plt.scatter(column1_list, column2_list, marker='o')
    # adding trend line
    z = np.polyfit(column1_list, column2_list, 1)
    p = np.poly1d(z)
    plt.plot(column1_list, p(column1_list), "r--")

    plt.xlim([9000, 70000])
    plt.ylim([7000, 80000])
    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# This function calculates the frequencies of a value in a given column
def column_frequencies_calculator(column):
    values_list = column.tolist()
    total_values = len(values_list)
    value_list = []  # A list where exist values will be stored once
    value_list_freq = []  # A list where the frequencies will be stored
    for value in values_list:
        if value not in value_list:
            value_list.append(value)
            value_list_freq.append((values_list.count(value)/total_values))
    return value_list_freq


# This function returns a list with all the values that exist in the original list, once
def extract_representing_values_from_list(lst):
    one_value_list = []
    for obj in lst:
        if obj not in one_value_list:
            one_value_list.append(obj)
    return one_value_list


# This function counts a specific value in a specific column
def len_df(dataframe, column, searched_value):
    return len(dataframe[dataframe[column] == searched_value])


#  This function counts how many times a given value is exists in a given list
def count(lst, value_to_count):
    return lst.count(value_to_count)


# This function create a heatmap of the pairs count of two given columns
def two_columns_frequencies_df(df_columns, column_name1, column_name2):
    pair_values_list = []
    new_dict = {}
    for i in df_columns.index:  # Iterating over the rows to create a list with all the pairs
        pair_values_list.append([df_columns[column_name1][i], df_columns[column_name2][i]])

    for i in df_columns.index:  # Iterating over the rows to create a dictionary with the pairs and their count
        column1_value = df_columns[column_name1][i]  # Extracting the first value of the pair
        column2_value = df_columns[column_name2][i]  # Extracting the second value of the pair
        value_count = count(pair_values_list, [column1_value, column2_value])  # Counting how many of each pair
        new_dict.update({(column1_value, column2_value): value_count})  # Updating the dictionary-key:pair, value:count

    first_tuple_element = []
    for key1 in new_dict.keys():
        if key1[0] not in first_tuple_element:
            first_tuple_element.append(key1[0])  # Creating a list with the morphologies of the first column
    second_tuple_element = []
    for key2 in new_dict.keys():
        if key2[1] not in second_tuple_element:
            second_tuple_element.append(key2[1])  # Creating a list with the morphologies of the second column

    new_df = pd.DataFrame(index=first_tuple_element, columns=second_tuple_element)  # df with the morphologies names

    for key, val in new_dict.items():  # Iterating over the dict to insert value for each df match key
        new_df.at[key[0], key[1]] = val
    new_df = new_df.fillna(0)
    return new_df


# Scatter matrix plot - one df
def scatter_matrix_plot(df_scatter_matrix, save_path=None):
    plt.clf()
    sns.pairplot(df_scatter_matrix)  # seaborn graph
    # scatter_matrix(df_scatter_matrix, figsize=(8, 8))  # matplotlib graph
    # plt.title('Scatter plot')
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


def mean_feature_vs_time(mean_list, time_points, title='', x_title='', y_title='', save_path=None):
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=20, color='black')
    plt.xlabel(x_title, fontsize=16)
    plt.ylabel(y_title, fontsize=16, rotation=0, labelpad=30)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    for mean in mean_list:
        plt.plot(time_points, mean, marker='o')

    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


def check_if_df_len_equals_to_excel_rows_number(data_frame, col_name):
    index_to_drop = []
    for index in data_frame.index:
        if data_frame.loc[index, col_name] == 0:
            index_to_drop.append(index)
    data_frame.drop(index_to_drop, inplace=True)
    return data_frame


# This function creates a beeswarm plot to chosen conditions and column
def beeswarm_graph(conditions_list, column_name, save_path):
    concat_data = pd.DataFrame()  # empty df
    con_group_number = 0  # conditions number counter
    for con_index in range(len(conditions_list)):
        temp = conditions_list[con_index]  # exporting specific condition
        temp_col = temp[column_name]  # exporting specific column
        temp_col = temp_col.reset_index()
        temp_col = temp_col.drop(['positions'], axis=1)
        index = list([con_group_number+1] * len(temp_col))  # creating a list with the condition number
        temp_col.index = index  # adding index
        concat_data = pd.concat([concat_data, temp_col], axis=0)  # concatenating the condition
        con_group_number += 1
    concat_data = concat_data.reset_index()
    concat_data = concat_data.rename(columns={'index': 'Condition'})
    updated_data = check_if_df_len_equals_to_excel_rows_number(concat_data, column_name)
    sns.swarmplot(x='Condition', y=column_name, data=updated_data)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


# This function creates a beesworm and a box plot graph on top, for chosen conditions and column
def beeswarm_box_plot_graph(conditions_list, conditions_num_list, column_name, save_path):
    concat_data = pd.DataFrame()  # empty df
    con_group_number = 0  # conditions number counter
    for con_index in range(len(conditions_list)):
        temp = conditions_list[con_index]  # exporting specific condition
        temp_col = temp[column_name]  # exporting specific column
        temp_col = temp_col.reset_index()
        temp_col = temp_col.drop(['positions'], axis=1)
        index = list([conditions_num_list[con_group_number]] * len(temp_col))  # creating a list with the con numbers
        temp_col.index = index  # adding index
        concat_data = pd.concat([concat_data, temp_col], axis=0)  # concatenating the condition
        con_group_number += 1
    concat_data = concat_data.reset_index()
    concat_data = concat_data.rename(columns={'index': 'Condition'})
    updated_data = check_if_df_len_equals_to_excel_rows_number(concat_data, column_name)
    order = conditions_num_list
    # plt.figure(figsize=(5, 7.5))
    ax = sns.boxplot(data=updated_data, x='Condition', y=column_name, showmeans=True,
                     meanprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "red"},
                     boxprops={'facecolor': 'None'})
    sns.swarmplot(data=updated_data, x='Condition', y=column_name, zorder=0.5)
    test_results = add_stat_annotation(ax, data=updated_data, x='Condition', y=column_name, order=order,
                                       box_pairs=[('1', '2'), ('1', '3'), ('2', '3'), ('1', '4'), ('2', '4'),
                                                  ('3', '4'), ('1', '5'), ('2', '5'), ('3', '5'), ('4', '5'),
                                                  ('1', '6'), ('2', '6'), ('3', '6'), ('4', '6'), ('5', '6'),
                                                  ('1', '7'), ('2', '7'), ('3', '7'), ('4', '7'), ('5', '7'),
                                                  ('6', '7'), ('1', '8'), ('2', '8'), ('3', '8'), ('4', '8'),
                                                  ('5', '8'), ('6', '8'), ('7', '8'), ('1', '9'), ('2', '9'),
                                                  ('3', '9'), ('4', '9'), ('5', '9'), ('6', '9'), ('7', '9'),
                                                  ('8', '9'), ('1', '10'), ('2', '10'), ('3', '10'), ('4', '10'),
                                                  ('5', '10'), ('6', '10'), ('7', '10'), ('8', '10'), ('9', '10')],
                                       test='t-test_ind', text_format='star')
    ax.set_facecolor('white')
    # ax.set_ylabel('Area')
    # ax.set_xlabel(None)
    if save_path is not None:
        # plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# This function creates a scatter plot that takes two groups and compare two measurements columns of them
def scatter_plot_compare_two_groups(data_frame, x_column_name, y_column_name, morphologies_column_name,
                                    morphology_name1, morphology_name2, title, save_path):
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.title(title, fontsize=20, color='black')
    plt.xlabel(x_column_name, fontsize=16)
    plt.ylabel(y_column_name, fontsize=16)

    morphology1_df = extract_columns_and_rows_to_new_df(data_frame, [x_column_name, y_column_name],
                                                        morphologies_column_name, morphology_name1)
    morphology2_df = extract_columns_and_rows_to_new_df(data_frame, [x_column_name, y_column_name],
                                                        morphologies_column_name, morphology_name2)

    plt.scatter(morphology1_df[x_column_name], morphology1_df[y_column_name], marker='o', color='red',
                label=morphology_name1)
    plt.scatter(morphology2_df[x_column_name], morphology2_df[y_column_name], marker='o', color='green',
                label=morphology_name2)
    plt.legend()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()


# This function creates a new data frame of specific morphology and specific columns
def extract_columns_and_rows_to_new_df(data_frame, columns_list, morphology_col, morphology_name):
    # extracting the columns
    new_columns = []
    for column in columns_list:
        new_columns.append(data_frame[column])
    new_data = pd.concat(new_columns, axis=1)

    # extracting the rows
    rows_to_drop = []
    for index in data_frame.index:
        if data_frame.loc[index, morphology_col] != morphology_name:
            rows_to_drop.append(index)
    new_data.drop(rows_to_drop, inplace=True)
    return new_data


def boxplot(conditions_list, column_name, conditions_num_list, save_path):
    concat_data = pd.DataFrame()  # empty df
    con_group_number = 0  # conditions number counter
    for con_index in range(len(conditions_list)):
        temp = conditions_list[con_index]  # exporting specific condition
        temp_col = temp[column_name]  # exporting specific column
        temp_col = temp_col.reset_index()
        temp_col = temp_col.drop(['positions'], axis=1)
        index = list([conditions_num_list[con_group_number]] * len(temp_col))  # creating a list with the con numbers
        temp_col.index = index  # adding index
        concat_data = pd.concat([concat_data, temp_col], axis=0)  # concatenating the condition
        con_group_number += 1
    concat_data = concat_data.reset_index()
    concat_data = concat_data.rename(columns={'index': 'Condition'})
    updated_data = check_if_df_len_equals_to_excel_rows_number(concat_data, column_name)
    ax = sns.boxplot(data=updated_data, x='Condition', y=column_name, showmeans=True,
                     meanprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "red"})

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()


# Change the index_range according to the length of your column
def adding_random_number_to_int(int_column, int_to_change):
    int_column_list = int_column.values.tolist()
    new_list = []
    for int_value in int_column_list:
        if int_value == int_to_change:
            int_value += np.random.uniform(0, 2)
        new_list.append(int_value)
    index_range = range(1, 121)
    new_list_column = pd.DataFrame(new_list, index=index_range)
    return new_list_column


def adding_int_random_number_to_number_in_a_list(int_list):
    changed_num_list = []
    a = randrange(0, 100)
    if a // 2:
        for num in int_list:
            num += np.random.uniform(0.5, 5.5)
            changed_num_list.append(num)
    else:
        for num in int_list:
            num -= np.random.uniform(0.5, 5.5)
            changed_num_list.append(num)
    return changed_num_list


def pca_plot(uniq_morph, pca_df, title='', save_path=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title(title, fontsize=20)
    targets = uniq_morph
    colors = ['r', 'g', 'b']
    for target, color in zip(targets, colors):
        indices_to_keep = pca_df['morphology'] == target
        ax.scatter(pca_df.loc[indices_to_keep, 'principal component 1'],
                   pca_df.loc[indices_to_keep, 'principal component 2'],
                   c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.tight_layout()
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
    plt.close()
