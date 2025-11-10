import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.utils.tf_utils import dataset_is_infinite
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix,classification_report
import random


data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


feat_info('mort_acc')

df = pd.read_csv('lending_club_loan_two.csv')
print(df.info())

# # Project Tasks
# # Section 1: Exploratory Data Analysis
#
# # 1) Since we will be attempting to predict loan_status, create a countplot
# sns.countplot(data=df,x='loan_status',palette='rainbow')
# plt.show()
#
# # 2) A histogram of the loan_amnt
# plt.figure(figsize=(12,4))
# sns.histplot(df['loan_amnt'],bins=40)
# plt.show()
#
# # 3) Calculate correlation between all continuous numeric variables
# dc = df.corr(numeric_only=True)
# print(dc)
#
# # and visualize this using a heatmap: Ι noticed that we have almost perfect correlation with the installment feature
# plt.figure(figsize=(12,7))
# sns.heatmap(data=dc,annot=True,cmap='viridis')
# plt.ylim(10,0)
# plt.show()
#
# # 4) printing out their descriptions and perform a scatterplot between them.
# feat_info('installment')
# feat_info('loan_amnt')
# sns.scatterplot(x='installment',y='loan_amnt',data=df)
# plt.show()
#
# # 5) create a boxplot between loan_status and Loan Amount.
# sns.boxplot(data=df,x='loan_status',y='loan_amnt',palette='rainbow')
# plt.show()
#
# # 6) Calculate the summary statistics for the loan amount, grouped by the loan_status.
# print(df.groupby('loan_status').describe())
#
# # 7) Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans.
# # What are the unique possible grades and subgrades?
# print(sorted(df['grade'].unique()))
# print("\n")
# print(sorted(df['sub_grade'].unique()))
#
# # 8) Create a countplot per grade. Set the hue to the loan_status label
# sns.countplot(data=df,x='grade',hue='loan_status')
# plt.show()
#
# # 9) Display a count plot per subgrade. You may need to resize for this plot
# # and [reorder](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot) the x axis. Feel free to edit the color palette. Explore both all loans made per subgrade as well being separated based on the loan_status. After creating this plot,
# # go ahead and create a similar plot, but set hue="loan_status"**
# plt.figure(figsize=(12,4))
# sorted_subs = sorted(df['sub_grade'].unique())
# sns.countplot(data=df,x='sub_grade',order=sorted_subs,palette='rainbow')
# plt.show()
#
# sns.countplot(data=df,x='sub_grade',order=sorted_subs,palette='rainbow',hue='loan_status')
# plt.show()
#
# # 10) It looks like F and G subgrades don't get paid back that often. Isolate those and recreate the countplot just for those subgrades.**
# plt.figure(figsize=(12,4))
# f_g = df[(df['grade'] == 'G') | (df['grade'] == 'F')]
# upgrade_order = sorted(f_g['sub_grade'].unique())
# sns.countplot(data=f_g,x='sub_grade',palette='coolwarm',order=upgrade_order,hue='loan_status')
# plt.show()
#
# # 11) Create a new column called 'loan_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".
#
def func(loan_status):
    if loan_status == 'Fully Paid':
        return 1
    else:
        return 0

df['loan_repaid'] = df['loan_status'].apply(func)
#
# # 12) CHALLENGE TASK: (Note this is hard, but can be done in one line!) Create a bar plot showing the correlation of the numeric features to the new loan_repaid column.
# sns.barplot(x=df.corr(numeric_only=True)['loan_repaid'].values,
#             y=df.corr(numeric_only=True)['loan_repaid'].index)
# plt.show()

# Section 2: Data PreProcessing
# Section Goals: Remove or fill any missing data. Remove unnecessary or repetitive features.
# Convert categorical string features to dummy variables.

# 1) What is the length of the dataframe?
print(df.shape[0])

# 2) Create a Series that displays the total count of missing values per column
missing_values = df.isnull().sum()
series = pd.Series(data=missing_values)
print(series)

# 3) Convert this Series to be in term of percentage of the total DataFrame
missing_percentage = (missing_values/len(df)) * 100
print(missing_percentage)

# 4) Let's examine emp_title and emp_length to see whether it will be okay to drop them. Print out their feature information using the feat_info() function from the top of this notebook.
feat_info('emp_title')
feat_info('emp_length')

# 5) How many unique employment job titles are there?
unique_jobs = df['emp_title'].nunique()
print(unique_jobs)
print("\n")
print(df['emp_title'].value_counts())

# 6) Let's remove that emp_title column.
df = df.drop('emp_title',axis=1)

# 7) Create a count plot of the emp_length feature column. Challenge: Sort the order of the values.
sorted(df['emp_length'].dropna().unique())
emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']
plt.figure(figsize=(12,4))
sns.countplot(data=df,x='emp_length',order=emp_length_order,palette='coolwarm')
plt.show()

# 8) Plot out the countplot with a hue separating Fully Paid vs Charged Off
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')

# 9) CHALLENGE TASK This still doesn't really inform us if there is a strong relationship between employment length and being charged off, what we want is the percentage of charge offs per category. Essentially informing us what percent of people per employment category didn't pay back their loan. There are a multitude of ways to create this Series. Once you've created it, see if visualize it with a [bar plot](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html). This may be tricky, refer to solutions if you get stuck on creating this Series.

emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp
print(emp_len)

# sns.barplot(data=df,x='e')
# plt.show()

# 10) Charge off rates are extremely similar across all employment lengths. Go ahead and drop the emp_length column.
df = df.drop('emp_length',axis=1)

# 11) Revisit the DataFrame to see what feature columns still have missing data.
missing_values = df.isnull().sum()
series = pd.Series(data=missing_values)
print(series)

# 12) Review the title column vs the purpose column. Is this repeated information? YESSSSS
print(df['purpose'].head(10))
print("\n")
print(df['title'].head(10))

# 13) The title column is simply a string subcategory/description of the purpose column. Go ahead and drop the title column.
df = df.drop('title',axis=1)

# 14) Find out what the mort_acc feature represents
feat_info('mort_acc')

# 15) Create a value_counts of the mort_acc column
val_counts = df['mort_acc'].value_counts()
print(val_counts)

# 16) There are many ways we could deal with this missing data. We could attempt to build a simple model to fill it in, such as a linear model, we could just fill it in based on the mean of the other columns, or you could even bin the columns into categories and then set NaN as its own category. There is no 100% correct approach! Let's review the other columsn to see which most highly correlates to mort_acc
print("Correlation with the mort_acc column")
print(df.corr(numeric_only=True)['mort_acc'].sort_values())

# 17) We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry.
print("Mean of mort_acc column per total_acc")
print(df.groupby('total_acc').mean(numeric_only=True)['mort_acc'])

# 18) CHALLENGE TASK: Let's fill in the missing mort_acc values based on their total_acc value. If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. This involves using an .apply() method with two columns. Check out the link below for more info, or review the solutions video/notebook.
total_acc_avg = df.groupby('total_acc').mean(numeric_only=True)['mort_acc']


def fill_mort_acc(total_acc, mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.

    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc

df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
print(df.isnull().sum())

# 19) revol_util and the pub_rec_bankruptcies have missing data points, but they account for less than 0.5% of the total data. Go ahead and remove the rows that are missing those values in those columns with dropna().
df = df.dropna()
print(df.isnull().sum())

#  Categorical Variables and Dummy Variables
# List all the columns that are currently non-numeric.
print(df.select_dtypes(include='object').columns)

# 20) Let's now go through all the string features to see what we should do with them.
# a) term feature : Convert the term feature into either a 36 or 60 integer numeric data type using .apply() or .map().
df['term'] = df['term'].map({'36 months': 36, '60 months': 60})

# b) grade feature: Drop it
df = df.drop('grade',axis=1)

# c) Convert the subgrade into dummy variables. Then concatenate these new columns to the original dataframe. Remember to drop the original subgrade column and to add drop_first=True to your get_dummies call.
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
print(df.columns)

print(df.select_dtypes(['object']).columns)

# d) verification_status, application_type,initial_list_status,purpose
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)

# e) home_ownership
print(df['home_ownership'].value_counts())

# Convert these to dummy variables, but [replace](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html) NONE and ANY with OTHER, so that we end up with just 4 categories, MORTGAGE, RENT, OWN, OTHER. Then concatenate them with the original dataframe. Remember to set drop_first=True and to drop the original columns.
df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)

# f) address :Let's feature engineer a zip code column from the address in the data set. Create a column called 'zip_code' that extracts the zip code from the address column. And make this zip_code into dummy variables using pandas
df['zip_code'] = df['address'].apply(lambda x: x.split()[-1])
zip_dummy = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop('zip_code',axis=1)

# Concatenate dummy variables back to main DataFrame
df = pd.concat([df, zip_dummy], axis=1)

# f) issue_d :
df = df.drop('issue_d',axis=1)

# g) earliest_cr_line
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)
df.select_dtypes(['object']).columns

# Train Test Split
# 1) drop the load_status column we created earlier, since its a duplicate of the loan_repaid column. We'll use the loan_repaid column since its already in 0s and 1s.
df = df.drop('loan_status',axis=1)

# 2) Set X and y variables to the .values of the features and label.
X = df.drop(['loan_repaid', 'address'], axis=1)
y = df['loan_repaid'].values

# Optional Step
df = df.sample(frac=0.1,random_state=101)
print(len(df))

# 3) Perform a train/test split with test_size=0.2 and a random_state of 101.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=101)

# Normalizing the Data with MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creating the Model
# Build a sequential model to will be trained on the data. You have unlimited options here, but here is what the solution uses: a model that goes 78 --> 39 --> 19--> 1 output neuron.
model = Sequential()
model.add(Dense(78,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(39,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(19,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

# Model Fit
model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=256,epochs=25)

# Model Save
model.save('full_data_project_model.h5')

# SECTION 3: EVALUATING MODEL PERFORMANCE a)
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
plt.show()

# b) Create predictions from X_test set and display classification report and confusion matrix for X_test set.
# Προβλέψεις (πιθανότητες)
predictions = model.predict(X_test)

# Μετατροπή σε 0/1
predictions = (predictions > 0.5).astype(int)
print(classification_report(y_test,predictions))
print("\n")
print(confusion_matrix(y_test,predictions))

## ΙTS NOT CORRECT
# c) Given the customer below, would you offer this person a loan?
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
print(new_customer)

model.predict(new_customer.values.reshape(1,78))
predictions = (predictions > 0.5).astype(int)

# d) Now check, did this person actually end up paying back their loan?
print(df.iloc[random_ind]['loan_repaid'])