#!/usr/bin/env python
# coding: utf-8

# # Overview

# 1. Project Instructions & Prerequisites
# 2. Learning Objectives
# 3. Data Preparation
# 4. Create Categorical Features with TF Feature Columns
# 5. Create Continuous/Numerical Features with TF Feature Columns
# 6. Build Deep Learning Regression Model with Sequential API and TF Probability Layers
# 7. Evaluating Potential Model Biases with Aequitas Toolkit
# 

# #  1. Project Instructions & Prerequisites

# ## Project Instructions

# **Context**: EHR data is becoming a key source of real-world evidence (RWE) for the pharmaceutical industry and regulators to [make decisions on clinical trials](https://www.fda.gov/news-events/speeches-fda-officials/breaking-down-barriers-between-clinical-trials-and-clinical-care-incorporating-real-world-evidence). You are a data scientist for an exciting unicorn healthcare startup that has created a groundbreaking diabetes drug that is ready for clinical trial testing. It is a very unique and sensitive drug that requires administering the drug over at least 5-7 days of time in the hospital with frequent monitoring/testing and patient medication adherence training with a mobile application. You have been provided a patient dataset from a client partner and are tasked with building a predictive model that can identify which type of patients the company should focus their efforts testing this drug on. Target patients are people that are likely to be in the hospital for this duration of time and will not incur significant additional costs for administering this drug to the patient and monitoring.  
# 
# In order to achieve your goal you must build a regression model that can predict the estimated hospitalization time for a patient and use this to select/filter patients for your study.
# 

# **Expected Hospitalization Time Regression Model:** Utilizing a synthetic dataset(denormalized at the line level augmentation) built off of the UCI Diabetes readmission dataset, students will build a regression model that predicts the expected days of hospitalization time and then convert this to a binary prediction of whether to include or exclude that patient from the clinical trial.
# 
# This project will demonstrate the importance of building the right data representation at the encounter level, with appropriate filtering and preprocessing/feature engineering of key medical code sets. This project will also require students to analyze and interpret their model for biases across key demographic groups. 
# 
# Please see the project rubric online for more details on the areas your project will be evaluated.

# ### Dataset

# Due to healthcare PHI regulations (HIPAA, HITECH), there are limited number of publicly available datasets and some datasets require training and approval. So, for the purpose of this exercise, we are using a dataset from UC Irvine(https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) that has been modified for this course. Please note that it is limited in its representation of some key features such as diagnosis codes which are usually an unordered list in 835s/837s (the HL7 standard interchange formats used for claims and remits).

# **Data Schema**
# The dataset reference information can be https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/project/data_schema_references/
# . There are two CSVs that provide more details on the fields and some of the mapped values.

# ## Project Submission 

# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "student_project_submission.ipynb" and save another copy as an HTML file by clicking "File" -> "Download as.."->"html". Include the "utils.py" and "student_utils.py" files in your submission. The student_utils.py should be where you put most of your code that you write and the summary and text explanations should be written inline in the notebook. Once you download these files, compress them into one zip file for submission.

# ## Prerequisites 

# - Intermediate level knowledge of Python
# - Basic knowledge of probability and statistics
# - Basic knowledge of machine learning concepts
# - Installation of Tensorflow 2.0 and other dependencies(conda environment.yml or virtualenv requirements.txt file provided)

# ## Environment Setup

# For step by step instructions on creating your environment, please go to https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/project/README.md.

# # 2.  Learning Objectives

# By the end of the project, you will be able to 
#    - Use the Tensorflow Dataset API to scalably extract, transform, and load datasets and build datasets aggregated at the line, encounter, and patient data levels(longitudinal)
#    - Analyze EHR datasets to check for common issues (data leakage, statistical properties, missing values, high cardinality) by performing exploratory data analysis.
#    - Create categorical features from Key Industry Code Sets (ICD, CPT, NDC) and reduce dimensionality for high cardinality features by using embeddings 
#    - Create derived features(bucketing, cross-features, embeddings) utilizing Tensorflow feature columns on both continuous and categorical input features
#    - SWBAT use the Tensorflow Probability library to train a model that provides uncertainty range predictions that allow for risk adjustment/prioritization and triaging of predictions
#    - Analyze and determine biases for a model for key demographic groups by evaluating performance metrics across groups by using the Aequitas framework 
# 

# # 3. Data Preparation

# In[559]:


# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import pandas as pd
import aequitas as ae
import tensorflow_data_validation as tfdv

# Put all of the helper functions in utils
from utils import build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable
pd.set_option('display.max_columns', 500)
# this allows you to make changes and save in student_utils.py and the file is reloaded every time you run a code block
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')


# In[560]:


#OPEN ISSUE ON MAC OSX for TF model training
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# ## Dataset Loading and Schema Review

# Load the dataset and view a sample of the dataset along with reviewing the schema reference files to gain a deeper understanding of the dataset. The dataset is located at the following path https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/project/starter_code/data/final_project_dataset.csv. Also, review the information found in the data schema https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/project/data_schema_references/

# In[561]:


dataset_path = "./data/final_project_dataset.csv"
df = pd.read_csv(dataset_path)


# In[562]:


df.head()


# ## Determine Level of Dataset (Line or Encounter)

# **Question 1**: Based off of analysis of the data, what level is this dataset? Is it at the line or encounter level? Are there any key fields besides the encounter_id and patient_nbr fields that we should use to aggregate on? Knowing this information will help inform us what level of aggregation is necessary for future steps and is a step that is often overlooked. 

# Student Response:??

# In[563]:


# The data set is at the line level. The number of lines in the data frame (143424) is greater than the number of 
# unique encounter IDs (101766)
# Besides the encounter_id and patient_nbr fields, we could also potentially aggregate on the primary diagnosis code 
# since that is a potential correlation that might be simpler to capture while also being effective.


# In[564]:


len(df)


# In[565]:


len(df['encounter_id'].unique())


# In[566]:


len(df['patient_nbr'].unique())


# ## Analyze Dataset

# **Question 2**: Utilizing the library of your choice (recommend Pandas and Seaborn or matplotlib though), perform exploratory data analysis on the dataset. In particular be sure to address the following questions:  
#     - a. Field(s) with high amount of missing/zero values
#     - b. Based off the frequency histogram for each numerical field, which numerical field(s) has/have a Gaussian(normal) distribution shape?
#     - c. Which field(s) have high cardinality and why (HINT: ndc_code is one feature)
#     - d. Please describe the demographic distributions in the dataset for the age and gender fields.
#     
# 

# **OPTIONAL**: Use the Tensorflow Data Validation and Analysis library to complete. 
# - The Tensorflow Data Validation and Analysis library(https://www.tensorflow.org/tfx/data_validation/get_started) is a useful tool for analyzing and summarizing dataset statistics. It is especially useful because it can scale to large datasets that do not fit into memory. 
# - Note that there are some bugs that are still being resolved with Chrome v80 and we have moved away from using this for the project. 

# **Student Response**: ??

# In[567]:


pd.DataFrame({'columns': df.columns, 
                        'percent_null': df.isnull().sum() * 100 / len(df), 
                        'percent_zero': df.isin([0]).sum() * 100 / len(df),
                        'Unknown_?': df.isin(['?']).sum() * 100 / len(df),
                        'Invalid': df.isin(['Unknown/Invalid', 'None']).sum() * 100 / len(df)
            } )


# In[568]:


np.isnan(df['time_in_hospital']).sum()


# In[569]:


# The high '0' values for number_outpatient, number_inpatient and number_emergency is understandable since 
# they are categorical fields and not numeric, so we will leave them as-is.

# The 'weight', 'payer_code' and 'medical_specialty' fields have very high '?' values, so let us remove them from the frame
# The 'max_glu_serum' and 'A1Cresult' fields have high 'None', so let us remove them from the frame 

# There are 5 entries in the 'gender' field that has 'Unknown/Invalid', so let us remove those rows


# In[570]:


df.pop('weight')
df.pop('payer_code')
df.pop('medical_specialty')
df.pop('max_glu_serum')
df.pop('A1Cresult')

invalid_indices = df[df['gender'].str.contains("Invalid")].index
df.drop(invalid_indices, inplace=True)
df['gender'].unique()


# In[571]:


df['change'].unique()


# In[572]:


len(df)


# In[573]:


df.head()


# In[574]:


numeric_feature_list = ['encounter_id', 'patient_nbr', 'admission_type_id', 'time_in_hospital', 
                          'num_lab_procedures', 'number_diagnoses', 'num_medications', 'num_procedures']
categorical_feature_list = ['race', 'gender', 'age','admission_type_id', 'discharge_disposition_id', 'admission_source_id', 
                            'primary_diagnosis_code', 'other_diagnosis_codes',
                            'number_outpatient', 'number_inpatient', 'number_emergency', 'ndc_code', 
                            'change', 'readmitted']


# In[575]:


df[numeric_feature_list].describe()


# In[576]:


# 'number_diagnosis' is manifesting a high standard deviation of ~20, so perhaps we should eliminate that field 
# from the model. But keeping it for now.


# In[577]:


# Plot histograms of numeric features
for i in numeric_feature_list:
    plt.figure(figsize=(10,6))
    #plt.hist(df[i],label=i)
    #plt.legend()
    #plt.show()


# In[578]:


# From the above histograms, the 'num_lab_procedures' and 'num_medication' fields exhibit a normal-like distribution
# From the SNS plots (a few cells down below), "age" is also normally distributed


# In[579]:


# Show cardinality of the categorical features 
pd.DataFrame({'columns': df[categorical_feature_list].columns, 
               'cardinality': df[categorical_feature_list].nunique() } )
# Diagnosis codes have high cardinality. Also, the 'other_diagnostic_codes' fields has a '|' separator 
# that is not parsed yet.


# In[580]:


df['gender'].unique()


# In[581]:


len(df[df['gender']=='Unknown/Invalid']['gender'])


# In[582]:


len(df['gender'])


# In[583]:


import seaborn as sns

df["age"]
#sns.boxplot(y=df['gender'])

sns.countplot(x="age", data=df)


# In[584]:


sns.countplot(x="age", hue="gender", data=df)


# In[585]:


######NOTE: The visualization will only display in Chrome browser. ########
#full_data_stats = tfdv.generate_statistics_from_csv(data_location='./data/final_project_dataset.csv') 
#tfdv.visualize_statistics(full_data_stats)


# In[ ]:





# ## Reduce Dimensionality of the NDC Code Feature

# **Question 3**: NDC codes are a common format to represent the wide variety of drugs that are prescribed for patient care in the United States. The challenge is that there are many codes that map to the same or similar drug. You are provided with the ndc drug lookup file https://github.com/udacity/nd320-c1-emr-data-starter/blob/master/project/data_schema_references/ndc_lookup_table.csv derived from the National Drug Codes List site(https://ndclist.com/). Please use this file to come up with a way to reduce the dimensionality of this field and create a new field in the dataset called "generic_drug_name" in the output dataframe. 

# In[586]:


#NDC code lookup file
ndc_code_path = "./medication_lookup_tables/final_ndc_lookup_table"
ndc_code_df = pd.read_csv(ndc_code_path)


# In[587]:


ndc_code_df


# In[588]:


#from student_utils import reduce_dimension_ndc
def reduce_dimension_ndc(df, ndc_code_df):
    #new_df = df.merge(ndc_code_df[['NDC_Code', 'Non-proprietary Name']], 
    #                  how="inner", left_on='ndc_code', right_on="NDC_Code")
    new_df = pd.merge(df, ndc_code_df[['Non-proprietary Name', 'NDC_Code']],how="left", 
                      left_on='ndc_code', right_on='NDC_Code')
    new_df.drop(columns=["NDC_Code"], inplace=True)
    new_df.rename(columns={"Non-proprietary Name": "generic_drug_name"}, inplace=True)
    new_df.nunique()
    reduce_dim_df = new_df
    return reduce_dim_df

reduce_dim_df = reduce_dimension_ndc(df, ndc_code_df)


# In[589]:


reduce_dim_df.shape


# In[590]:


df.shape


# In[591]:


df.nunique()


# In[592]:


reduce_dim_df.nunique()


# In[593]:


# Number of unique values should be less for the new output field
assert df['ndc_code'].nunique() > reduce_dim_df['generic_drug_name'].nunique()


# ## Select First Encounter for each Patient 

# **Question 4**: In order to simplify the aggregation of data for the model, we will only select the first encounter for each patient in the dataset. This is to reduce the risk of data leakage of future patient encounters and to reduce complexity of the data transformation and modeling steps. We will assume that sorting in numerical order on the encounter_id provides the time horizon for determining which encounters come before and after another.

# In[594]:


#from student_utils import select_first_encounter

def select_first_encounter (reduce_dim_df):
    df = reduce_dim_df.sort_values('encounter_id')
    first_encounter_dataframe = df.groupby(['patient_nbr']).head(1)
    return first_encounter_dataframe

#print (len(df))
#print (len(first_encounter_df))
#print (df['patient_nbr'].nunique())
#print (first_encounter['patient_nbr'].nunique())
#print (first_encounter['encounter_id'].nunique())

first_encounter_df = select_first_encounter(reduce_dim_df)


# In[595]:


#first_encounter_df


# In[596]:


# unique patients in transformed dataset
unique_patients = first_encounter_df['patient_nbr'].nunique()
print("Number of unique patients:{}".format(unique_patients))

# unique encounters in transformed dataset
unique_encounters = first_encounter_df['encounter_id'].nunique()
print("Number of unique encounters:{}".format(unique_encounters))

original_unique_patient_number = reduce_dim_df['patient_nbr'].nunique()
# number of unique patients should be equal to the number of unique encounters and patients in the final dataset
assert original_unique_patient_number == unique_patients
assert original_unique_patient_number == unique_encounters
print("Tests passed!!")


# ## Aggregate Dataset to Right Level for Modeling 

# In order to provide a broad scope of the steps and to prevent students from getting stuck with data transformations, we have selected the aggregation columns and provided a function to build the dataset at the appropriate level. The 'aggregate_dataset" function that you can find in the 'utils.py' file can take the preceding dataframe with the 'generic_drug_name' field and transform the data appropriately for the project. 
# 
# To make it simpler for students, we are creating dummy columns for each unique generic drug name and adding those are input features to the model. There are other options for data representation but this is out of scope for the time constraints of the course.

# In[597]:


exclusion_list = ['generic_drug_name']
grouping_field_list = [c for c in first_encounter_df.columns if c not in exclusion_list]
agg_drug_df, ndc_col_list = aggregate_dataset(first_encounter_df, grouping_field_list, 'generic_drug_name')


# In[598]:


assert len(agg_drug_df) == agg_drug_df['patient_nbr'].nunique() == agg_drug_df['encounter_id'].nunique()


# In[599]:


agg_drug_df


# ## Prepare Fields and Cast Dataset 

# ### Feature Selection

# **Question 5**: After you have aggregated the dataset to the right level, we can do feature selection (we will include the ndc_col_list, dummy column features too). In the block below, please select the categorical and numerical features that you will use for the model, so that we can create a dataset subset. 
# 
# For the payer_code and weight fields, please provide whether you think we should include/exclude the field in our model and give a justification/rationale for this based off of the statistics of the data. Feel free to use visualizations or summary statistics to support your choice.

# Student response: ??

# In[600]:


# The weight, payer_code and medical_specialty fields have very high '?' values and so they should be 
# exlcuded from our model. We have already dropped them from the dataframe during preprocessing
# Similarly A1Cresult and max_glu_serum have high 'None' values, so they have also been excluded. 
# See evidence for the above in the data processing code above

'''
Please update the list to include the features you think are appropriate for the model 
and the field that we will be using to train the model. There are three required demographic features for the model 
and I have inserted a list with them already in the categorical list. 
These will be required for later steps when analyzing data splits and model biases.
'''
required_demo_col_list = ['race', 'gender', 'age']
#student_categorical_col_list = ['primary_diagnosis_code', 'number_emergency', 'readmitted', 'change' ] + required_demo_col_list + ndc_col_list
student_categorical_col_list = [ 'readmitted', 'admission_type_id', 'discharge_disposition_id', 
                                'admission_source_id', 'primary_diagnosis_code', #'other_diagnosis_codes', 'ndc_code',
                               ] + required_demo_col_list + ndc_col_list
#student_numerical_col_list = ['num_lab_procedures', 'number_diagnoses', 'num_medications', 
#                               'num_procedures']
student_numerical_col_list = ['number_diagnoses', 'num_medications', 'num_procedures']
PREDICTOR_FIELD = 'time_in_hospital'


# In[601]:


def select_model_features(df, categorical_col_list, numerical_col_list, PREDICTOR_FIELD, grouping_key='patient_nbr'):
    selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list + numerical_col_list   
    return agg_drug_df[selected_col_list]


# In[602]:


selected_features_df = select_model_features(agg_drug_df, student_categorical_col_list, student_numerical_col_list,
                                            PREDICTOR_FIELD)


# In[603]:


selected_features_df.head()


# ### Preprocess Dataset - Casting and Imputing  

# We will cast and impute the dataset before splitting so that we do not have to repeat these steps across the splits in the next step. For imputing, there can be deeper analysis into which features to impute and how to impute but for the sake of time, we are taking a general strategy of imputing zero for only numerical features. 
# 
# OPTIONAL: What are some potential issues with this approach? Can you recommend a better way and also implement it?

# In[604]:


processed_df = preprocess_df(selected_features_df, student_categorical_col_list, 
        student_numerical_col_list, PREDICTOR_FIELD, categorical_impute_value='nan', numerical_impute_value=0)


# ## Split Dataset into Train, Validation, and Test Partitions

# **Question 6**: In order to prepare the data for being trained and evaluated by a deep learning model, we will split the dataset into three partitions, with the validation partition used for optimizing the model hyperparameters during training. One of the key parts is that we need to be sure that the data does not accidently leak across partitions.
# 
# Please complete the function below to split the input dataset into three partitions(train, validation, test) with the following requirements.
# - Approximately 60%/20%/20%  train/validation/test split
# - Randomly sample different patients into each data partition
# - **IMPORTANT** Make sure that a patient's data is not in more than one partition, so that we can avoid possible data leakage.
# - Make sure that the total number of unique patients across the splits is equal to the total number of unique patients in the original dataset
# - Total number of rows in original dataset = sum of rows across all three dataset partitions

# In[605]:


#from student_utils import patient_dataset_splitter
def patient_dataset_splitter (df, colname):
    train_dataset     = df.sample(frac=0.6,random_state=0)
    remaining_dataset = df.drop(train_dataset.index)
    valid_dataset     = remaining_dataset.sample(frac=0.5, random_state=0)
    test_dataset      = remaining_dataset.drop(valid_dataset.index)    
    return train_dataset, valid_dataset, test_dataset

d_train, d_val, d_test = patient_dataset_splitter(processed_df, 'patient_nbr')


# In[606]:


assert len(d_train) + len(d_val) + len(d_test) == len(processed_df)
print("Test passed for number of total rows equal!")


# In[607]:


assert (d_train['patient_nbr'].nunique() + d_val['patient_nbr'].nunique() + d_test['patient_nbr'].nunique()) == agg_drug_df['patient_nbr'].nunique()
print("Test passed for number of unique patients being equal!")


# ## Demographic Representation Analysis of Split

# After the split, we should check to see the distribution of key features/groups and make sure that there is representative samples across the partitions. The show_group_stats_viz function in the utils.py file can be used to group and visualize different groups and dataframe partitions.

# ### Label Distribution Across Partitions

# Below you can see the distributution of the label across your splits. Are the histogram distribution shapes similar across partitions?

# In[608]:


show_group_stats_viz(processed_df, PREDICTOR_FIELD)


# In[609]:


show_group_stats_viz(d_train, PREDICTOR_FIELD)


# In[610]:


show_group_stats_viz(d_test, PREDICTOR_FIELD)


# ## Demographic Group Analysis

# We should check that our partitions/splits of the dataset are similar in terms of their demographic profiles. Below you can see how we might visualize and analyze the full dataset vs. the partitions.

# In[611]:


# Full dataset before splitting
patient_demo_features = ['race', 'gender', 'age', 'patient_nbr']
patient_group_analysis_df = processed_df[patient_demo_features].groupby('patient_nbr').head(1).reset_index(drop=True)
show_group_stats_viz(patient_group_analysis_df, 'gender')


# In[612]:


# Training partition
show_group_stats_viz(d_train, 'gender')


# In[613]:


# Test partition
show_group_stats_viz(d_test, 'gender')


# ## Convert Dataset Splits to TF Dataset

# We have provided you the function to convert the Pandas dataframe to TF tensors using the TF Dataset API. 
# Please note that this is not a scalable method and for larger datasets, the 'make_csv_dataset' method is recommended -https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset.

# In[614]:


# Convert dataset from Pandas dataframes to TF dataset 
batch_size = 128
diabetes_train_ds = df_to_dataset(d_train, PREDICTOR_FIELD, batch_size=batch_size)
diabetes_val_ds = df_to_dataset(d_val, PREDICTOR_FIELD, batch_size=batch_size)
diabetes_test_ds = df_to_dataset(d_test, PREDICTOR_FIELD, batch_size=batch_size)


# In[615]:


# We use this sample of the dataset to show transformations later
diabetes_batch = next(iter(diabetes_train_ds))[0]
def demo(feature_column, example_batch):
    feature_layer = layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch))


# # 4. Create Categorical Features with TF Feature Columns

# ## Build Vocabulary for Categorical Features

# Before we can create the TF categorical features, we must first create the vocab files with the unique values for a given field that are from the **training** dataset. Below we have provided a function that you can use that only requires providing the pandas train dataset partition and the list of the categorical columns in a list format. The output variable 'vocab_file_list' will be a list of the file paths that can be used in the next step for creating the categorical features.

# In[616]:


vocab_file_list = build_vocab_files(d_train, student_categorical_col_list)


# In[617]:


#vocab_file_list


# ## Create Categorical Features with Tensorflow Feature Column API

# **Question 7**: Using the vocab file list from above that was derived fromt the features you selected earlier, please create categorical features with the Tensorflow Feature Column API, https://www.tensorflow.org/api_docs/python/tf/feature_column. Below is a function to help guide you.

# In[618]:


#from student_utils import create_tf_categorical_feature_cols
def create_tf_categorical_feature_cols (cat_col_list):
    one_hot_feature_list = []
    for i in range (0, len(cat_col_list)):                
        vocab = tf.feature_column.categorical_column_with_vocabulary_file(key=cat_col_list[i], 
                                                                          vocabulary_file = vocab_file_list[i], 
                                                                          num_oov_buckets=1)
        one_hot_feature = tf.feature_column.indicator_column(vocab)
        one_hot_feature_list.append(one_hot_feature)
    return one_hot_feature_list
    
tf_cat_col_list = create_tf_categorical_feature_cols(student_categorical_col_list)


# In[619]:


test_cat_var1 = tf_cat_col_list[0]
print("Example categorical field:\n{}".format(test_cat_var1))
demo(test_cat_var1, diabetes_batch)


# # 5. Create Numerical Features with TF Feature Columns

# **Question 8**: Using the TF Feature Column API(https://www.tensorflow.org/api_docs/python/tf/feature_column/), please create normalized Tensorflow numeric features for the model. Try to use the z-score normalizer function below to help as well as the 'calculate_stats_from_train_data' function.

# In[620]:


#from student_utils import create_tf_numeric_feature
import functools
def normalize_numeric_with_zscore(col, mean, std):
    return (col - mean)/std
    
def create_tf_numeric_feature (colname, mean_val, stdev_val):
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=mean_val, std=stdev_val)
    return tf.feature_column.numeric_column(key=colname, default_value = 0, 
                                            normalizer_fn=normalizer, dtype=tf.float64)


# For simplicity the create_tf_numerical_feature_cols function below uses the same normalizer function across all features(z-score normalization) but if you have time feel free to analyze and adapt the normalizer based off the statistical distributions. You may find this as a good resource in determining which transformation fits best for the data https://developers.google.com/machine-learning/data-prep/transform/normalization.
# 

# In[621]:


def calculate_stats_from_train_data(df, col):
    mean = df[col].describe()['mean']
    std = df[col].describe()['std']
    return mean, std

def create_tf_numerical_feature_cols(numerical_col_list, train_df):
    tf_numeric_col_list = []
    for c in numerical_col_list:
        mean, std = calculate_stats_from_train_data(train_df, c)
        tf_numeric_feature = create_tf_numeric_feature(c, mean, std)
        tf_numeric_col_list.append(tf_numeric_feature)
    return tf_numeric_col_list


# In[622]:


#print (student_numerical_col_list)
#print (student_numerical_col_list[0])
#d_train[student_numerical_col_list[0]]
tf_cont_col_list = create_tf_numerical_feature_cols(student_numerical_col_list, d_train)


# In[623]:


test_cont_var1 = tf_cont_col_list[0]
print("Example continuous field:\n{}\n".format(test_cont_var1))
demo(test_cont_var1, diabetes_batch)


# # 6. Build Deep Learning Regression Model with Sequential API and TF Probability Layers

# ## Use DenseFeatures to combine features for model

# Now that we have prepared categorical and numerical features using Tensorflow's Feature Column API, we can combine them into a dense vector representation for the model. Below we will create this new input layer, which we will call 'claim_feature_layer'.

# In[624]:


claim_feature_columns = tf_cat_col_list + tf_cont_col_list
claim_feature_layer = tf.keras.layers.DenseFeatures(claim_feature_columns)


# ## Build Sequential API Model from DenseFeatures and TF Probability Layers

# Below we have provided some boilerplate code for building a model that connects the Sequential API, DenseFeatures, and Tensorflow Probability layers into a deep learning model. There are many opportunities to further optimize and explore different architectures through benchmarking and testing approaches in various research papers, loss and evaluation metrics, learning curves, hyperparameter tuning, TF probability layers, etc. Feel free to modify and explore as you wish.

# **OPTIONAL**: Come up with a more optimal neural network architecture and hyperparameters. Share the process in discovering the architecture and hyperparameters.

# In[625]:


def build_sequential_model(feature_layer):
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dense(75, activation='relu'),
        tfp.layers.DenseVariational(1+1, posterior_mean_field, prior_trainable),
        tfp.layers.DistributionLambda(
            lambda t:tfp.distributions.Normal(loc=t[..., :1],
                                             scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])
                                             )
        ),
    ])
    return model

def build_diabetes_model(train_ds, val_ds,  feature_layer,  epochs=5, loss_metric='mse'):
    model = build_sequential_model(feature_layer)
    model.compile(optimizer='rmsprop', loss=loss_metric, metrics=[loss_metric])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=loss_metric, patience=3)     
    history = model.fit(train_ds, validation_data=val_ds,
                        callbacks=[early_stop],
                        epochs=epochs)
    return model, history 


# In[626]:


diabetes_model, history = build_diabetes_model(diabetes_train_ds, diabetes_val_ds,  claim_feature_layer,  epochs=10)


# ## Show Model Uncertainty Range with TF Probability

# **Question 9**: Now that we have trained a model with TF Probability layers, we can extract the mean and standard deviation for each prediction. Please fill in the answer for the m and s variables below. The code for getting the predictions is provided for you below.

# In[627]:


feature_list = student_categorical_col_list + student_numerical_col_list
diabetes_x_tst = dict(d_test[feature_list])
#d_test[feature_list].head()


# In[628]:


#print (diabetes_x_tst)
diabetes_yhat = diabetes_model(diabetes_x_tst)
preds = diabetes_model.predict(diabetes_test_ds)
print(diabetes_yhat)


# In[629]:


#from student_utils import get_mean_std_from_preds

def get_mean_std_from_preds (diabetes_yhat):
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

m, s = get_mean_std_from_preds(diabetes_yhat)


# In[630]:


print (m)


# ## Show Prediction Output 

# In[631]:


prob_outputs = {
    "pred": preds.flatten(),
    "actual_value": d_test['time_in_hospital'].values,
    "pred_mean": m.numpy().flatten(),
    "pred_std": s.numpy().flatten()
}
prob_output_df = pd.DataFrame(prob_outputs)


# In[632]:


prob_output_df.describe()


# ## Convert Regression Output to Classification Output for Patient Selection

# **Question 10**: Given the output predictions, convert it to a binary label for whether the patient meets the time criteria or does not (HINT: use the mean prediction numpy array). The expected output is a numpy array with a 1 or 0 based off if the prediction meets or doesnt meet the criteria.

# In[633]:


#from student_utils import get_student_binary_prediction

def get_student_binary_prediction(df, colname):
    # Trial and error based on (1) the output of prob_output_df.describe() above and (2) observed performance below
    return df[colname].apply(lambda x:1 if x>=6 else 0)

student_binary_prediction = get_student_binary_prediction(prob_output_df, 'pred_mean')


# ### Add Binary Prediction to Test Dataframe

# Using the student_binary_prediction output that is a numpy array with binary labels, we can use this to add to a dataframe to better visualize and also to prepare the data for the Aequitas toolkit. The Aequitas toolkit requires that the predictions be mapped to a binary label for the predictions (called 'score' field) and the actual value (called 'label_value').

# In[634]:


def add_pred_to_test(test_df, pred_np, demo_col_list):
    for c in demo_col_list:
        test_df[c] = test_df[c].astype(str)
    test_df['score'] = pred_np
    test_df['label_value'] = test_df['time_in_hospital'].apply(lambda x: 1 if x >=5 else 0)
    return test_df

pred_test_df = add_pred_to_test(d_test, student_binary_prediction, ['race', 'gender'])


# In[635]:


pred_test_df[['patient_nbr', 'gender', 'race', 'time_in_hospital', 'score', 'label_value']].head()
print ("# Match =", len(pred_test_df[(pred_test_df['score'] == pred_test_df['label_value'])]))
print ("# Mismatch =", len(pred_test_df[(pred_test_df['score'] != pred_test_df['label_value'])]))
print ("# NaNs in Prediction =", len(pred_test_df[np.isnan(pred_test_df['score'])]))


# ## Model Evaluation Metrics 

# **Question 11**: Now it is time to use the newly created binary labels in the 'pred_test_df' dataframe to evaluate the model with some common classification metrics. Please create a report summary of the performance of the model and be sure to give the ROC AUC, F1 score(weighted), class precision and recall scores. 

# For the report please be sure to include the following three parts:
# - With a non-technical audience in mind, explain the precision-recall tradeoff in regard to how you have optimized your model.
# 
# - What are some areas of improvement for future iterations?

# In[638]:


# AUC, F1, precision and recall
# Summary
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

y_true = pred_test_df['label_value'].values 
y_pred = pred_test_df['score'].values

#accuracy_score(y_true, y_pred)
#print(classification_report(y_true, y_pred))


# # 7. Evaluating Potential Model Biases with Aequitas Toolkit

# ## Prepare Data For Aequitas Bias Toolkit 

# Using the gender and race fields, we will prepare the data for the Aequitas Toolkit.

# In[ ]:


# Aequitas
from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.plotting import Plot
from aequitas.bias import Bias
from aequitas.fairness import Fairness

ae_subset_df = pred_test_df[['race', 'gender', 'score', 'label_value']]
ae_df, _ = preprocess_input_df(ae_subset_df)
g = Group()
xtab, _ = g.get_crosstabs(ae_df)
absolute_metrics = g.list_absolute_metrics(xtab)
clean_xtab = xtab.fillna(-1)
aqp = Plot()
b = Bias()


# ## Reference Group Selection

# Below we have chosen the reference group for our analysis but feel free to select another one.

# In[ ]:


# test reference group with Caucasian Male
bdf = b.get_disparity_predefined_groups(clean_xtab, 
                    original_df=ae_df, 
                    ref_groups_dict={'race':'Caucasian', 'gender':'Male'
                                     }, 
                    alpha=0.05, 
                    check_significance=False)


f = Fairness()
fdf = f.get_group_value_fairness(bdf)


# ## Race and Gender Bias Analysis for Patient Selection

# **Question 12**: For the gender and race fields, please plot two metrics that are important for patient selection below and state whether there is a significant bias in your model across any of the groups along with justification for your statement.

# In[ ]:


# Plot two metrics

# Is there significant bias in your model for either race or gender?


# ## Fairness Analysis Example - Relative to a Reference Group 

# **Question 13**: Earlier we defined our reference group and then calculated disparity metrics relative to this grouping. Please provide a visualization of the fairness evaluation for this reference group and analyze whether there is disparity.

# In[ ]:


# Reference group fairness plot

