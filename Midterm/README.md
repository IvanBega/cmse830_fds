# Midterm Project ReadMe
### Why I chose this dataset
I chose the "Heart Disease" and "Heart Disease Spread by Country" datasets because I want to explore in details what lifestyle habits can contribute to this disease and what can we do to prevent it. Unfortunately, according to World Health Organization (WHO), heart-related diseases remain the number one cause of death around the world. 

The primary reason I decided to go with the dataset is due to its simple yet rich features, such as age, BMI, stress level, sugar consumption, vital blood parameters, quality of sleep. They seem attractive to me because nearly every person can quickly determine to what "bucket" they belong without performing expensive medical assessments like MRI, X-Ray, etc. My dataset will try to answer the question of how likely it is for a person to develop heart diseases given their life circumstances.

### What have I learned from IDA/EDA

First, both datasets are almost complete and have very little missing data. They contain numerical, categorical, binary, and ordinal features which need to be handled accordingly. I have noticed that "Heart Disease" dataset is undersampled with respect to people who have reported heart disease.
### What preprocessing steps I've completed

- Handling missing values: imputation using KNN with n = 5
- Balancing undersampled records who have reported heart disease using SMOTE method from imblearn library
- Encoding binary variables with LabelEncoder, encoding ordinal variables with OrdinalEncoder
### What I've tried with Streamlit so far
- Pie plot
- Scatter plot
- Strip plot
- Choropleth map