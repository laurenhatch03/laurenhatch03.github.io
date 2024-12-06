
<img width="1395" alt="cover" src="https://github.com/user-attachments/assets/886add33-86f3-4481-8a91-510b7822cae5">


**A Report on Career Change Predictions**

**Atmospheric and Oceanic Science C111 Final Project** 

**Dr. Alexander Lozinski**

**December 6, 2024**

For this project, I used machine learning techniques to analyze the factors affecting whether an individual wants to change their career. I used this to try to predict whether an individual is likely to change their occupation based on those variables.
## Introduction 

Many people go to college, get a degree in a field they're interested in, and move on to get a job unrelated to their field of study. The question arises: does your field of study determine your career path? In this report, you will see the different variables that could lead to someone wanting to change their occupation.

Using the “Field of Study vs Occupation” dataset on Kaggle, I looked at variables that I believed would contribute to an individual, ages 20-30, wanting to change their occupation. I decided to use supervised learning because I wanted to make predictions based on data patterns between the relationships of variables. This is a classification problem, so supervised learning would be the best option. I modeled these variables with bar graphs to get an initial visual representation of the data. Next, I made a correlation matrix to find the variables that contribute most to individuals who are more likely to change occupations. Then, I set, split, and called the data to make my models. I created a decision tree classifier, logistic regression, and random forest classifier models. I then made REC curves for all three models to show how well they can predict outcomes. I then made confusion matrices to show how well these models can predict by summarizing the number of correct and incorrect predictions. I also showed the ROC curves to go with each model. I made decision trees to follow and show what values within variables lead to whether an individual is likely to change their occupation. Lastly, to confirm the correct variable was being used, I plotted the feature importances to show which variables each model claimed to contribute the most to these predictions.

## Data
[Click here to view the dataset I used!](https://www.kaggle.com/datasets/jahnavipaliwal/field-of-study-vs-occupation/data)

The dataset includes 38,444 rows, each representing a different person. I narrowed it down to ages 20-30 (10,525 rows) to see how it affects people closer to me. With 22 columns, each attribute contributes to different information about each person. These 22 columns include their field of study, current occupation, age, gender, years of experience, education level, industry growth rate, job satisfaction, work-life balance, job opportunities, salary, job security, career change interest, skills gap, family influence, mentorship available, certifications, freelancing experience, geographic mobility, professional networks, career change events, technology adoption, and likely to change occupation. The website details each and how the numbers are inputted into the dataset file. For this project, I used the variable 'Likely to Change Occupation' as my dependent variable to see what independent variables affect that likelihood.

## Preprocessing Steps

The first thing I had to do was upload the dataset into Google Colab. Next, I noticed the data set was massive, so I narrowed it down to ages 20-30 (10,525 rows) to better represent how it affects people closer to me. I then checked the data types to make sure I could use them.  After that, I had to convert the variables that were strings into integers so the data could be graphed. These variables included gender, family influence, field of study, occupation, education level, and industry growth rate. Using ```data.head()``` I checked that all the variables were integers and then proceeded to use ```data.describe()``` to look at averages.

Below is a snippet of the code where I cleaned up the data.
```python
#cutting data so it's only ages 20-30
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data = df[df['Age'] <= 30].copy()

#removing years of experience because it doesnt make sense
data.drop('Years of Experience', axis=1, inplace=True)

#converting strings to ints
gender_map = { 'Male': 0,  'Female': 1,}
family_influence_map = { np.nan : 0, 'Low': 1, 'Medium': 2, 'High': 3,} 
field_of_study_map = {'Education':0, 'Arts':1, 'Business':2, 'Law':3, 'Computer Science':4, 'Biology':5, 
                     'Mechanical Engineering':6, 'Economics':7,'Medicine':8, 'Psychology':9 }
occupation_map = {'Biologist':0, 'Doctor':1, 'Software Developer':2, 'Business Analyst':3, 'Economist':4,
                  'Mechanical Engineer':5, 'Artist':6, 'Psychologist':7, 'Teacher':8, 'Lawyer':9}
education_map = { 'High School': 0,"Bachelor's": 1, "Master's": 2, 'PhD': 3,}
industry_map = {'Low': 0, 'Medium': 1, 'High': 2,}

#mapping new variables
data['Gender'] = data['Gender'].map(gender_map)
data['Gender'] = data['Gender'].astype(int)
data['Family Influence'] = data['Family Influence'].map(family_influence_map)
data['Field of Study'] = data['Field of Study'].map(field_of_study_map)
data['Current Occupation'] = data['Current Occupation'].map(occupation_map)
data['Education Level'] = data['Education Level'].map(education_map)
data['Industry Growth Rate'] = data['Industry Growth Rate'].map(industry_map)
data.head(5)

#checks that all variables are integers
print(data.dtypes)
```
Below are the five rows that were printed out with ``` data.head(5) ``` that show all variables with integer values.

<img width="1261" alt="Screenshot 2024-12-03 at 7 22 32 PM" src="https://github.com/user-attachments/assets/a69a7c6e-0d8e-4aa8-989f-24e8f9795051">

<img width="721" alt="Screenshot 2024-12-03 at 7 23 34 PM" src="https://github.com/user-attachments/assets/fc0af86f-0088-4e75-ba02-e41b9772bea3">

Using ```data.describe() ```, I was able to see averages of every variable; they are shown below.

<img width="1235" alt="Screenshot 2024-12-03 at 7 24 47 PM" src="https://github.com/user-attachments/assets/3febe9d6-c0bf-4cc0-8e7a-cc2fd2558e1c">

<img width="1182" alt="Screenshot 2024-12-03 at 7 26 16 PM" src="https://github.com/user-attachments/assets/c96a4736-943a-456b-aedb-5d37ff65e651">


Next, I made bar graphs for each variable with ‘Likely to Change Occupation’ as the dependent variable. The graphs are shown below:


<img width="550" alt="figure 1" src="https://github.com/user-attachments/assets/48ba93b5-fa68-4772-ac69-ef349a555dd9">  

*Figure 1: Occupation Change by Field of Study* 

- In Figure 1, when looking at the fields of study modeled vs. the likelihood to change occupations, no one significantly stands out. This means that an individual's field of study doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 2" src="https://github.com/user-attachments/assets/10adbf45-904c-4c26-b6fb-07ead31b21cc">

*Figure 2: Occupation Change by Current Occupation* 

- In Figure 2, when looking at current occupation vs. likelihood to change occupation, no one significantly stands out. This means that an individual's current occupation doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 3" src="https://github.com/user-attachments/assets/12d7ebab-0952-481d-8b4a-a5ae3e8dceaa">

*Figure 3: Occupation Change by Age* 

- In Figure 3, when looking at age vs. the likelihood to change occupation, no one significantly stands out. This means that an individual's age doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 4" src="https://github.com/user-attachments/assets/050840dd-4d52-4974-843a-ba6fa3b0fa7c">  

*Figure 4: Occupation Change by Gender*

- In Figure 4, when looking at gender vs. the likelihood to change occupation, no gender significantly stands out. This means that an individual's gender doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 5" src="https://github.com/user-attachments/assets/4a5919ed-31f7-4b6e-a1ec-7ede7fe5ddf3">

*Figure 5: Occupation Change by Education Level*

- In Figure 5, when looking at education level vs. the likelihood to change occupation, no one significantly stands out. This means that an individual's education level doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 6" src="https://github.com/user-attachments/assets/57eb8c4d-d504-47bd-a0d1-c1451e531017">

*Figure 6: Occupation Change by Industry Growth Rate*

- In Figure 6, individuals in an industry with a higher growth rate are more likely to change their occupation. Although there is a slight correlation, the difference between low, medium, and high isn't significant enough to differentiate between them.

<img width="550" alt="figure 7" src="https://github.com/user-attachments/assets/c93f087c-72b2-45f7-a4cf-f31e6a00a76f">  

*Figure 7: Occupation Change by Job Satisfaction*

- In Figure 7, individuals who rated their job satisfaction as four or lower on a 1-10 scale are almost certain to change their job satisfaction. This makes sense because if you aren’t enjoying your job, you want to find a different one.

<img width="550" alt="figure 8" src="https://github.com/user-attachments/assets/6298dcf0-2653-46da-9f1d-e3ed86ba4323">

*Figure 8: Occupation Change by Work-Life Balance*

- In Figure 8, no one stands out significantly when comparing work-life balance to the likelihood of changing occupations. The likelihood of changing occupations fluctuates, so an individual's rating of work-life balance on a 1-10 scale doesn’t directly correlate to changing occupations.

<img width="550" alt="figure 9" src="https://github.com/user-attachments/assets/374dcefa-d847-4fc9-b406-fd2cebd7de36">

*Figure 9: Occupation Change by Job Opportunities*

- In Figure 9, when looking at job opportunities vs. the likelihood to change occupation, no one significantly stands out. This means that an individual's number of job opportunities doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 10" src="https://github.com/user-attachments/assets/6b6e5296-f101-4979-b839-f02b61223e7b"> 

*Figure 10: Occupation Change by Salary*

- Figure 10 shows a graph showing that individuals who make between $30k and $60k are much more likely to change occupations than someone who makes more than that.

<img width="550" alt="figure 11" src="https://github.com/user-attachments/assets/e7858780-aff9-4f70-980f-e65f96bcadc7"> 

*Figure 11: Occupation Change by Job Security*

- In Figure 11, no one stands out significantly when looking at job security vs. the likelihood to change occupation. This means that an individual's rating of job security, on a 1-10 scale, doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 12" src="https://github.com/user-attachments/assets/98bc2f7d-dcb2-45e0-a434-8b29ec6c938f">

*Figure 12: Occupation Change by Career Change Interest*

- In Figure 12, as one can expect, those interested in changing their career are more likely to change occupations.
  
<img width="550" alt="figure 13" src="https://github.com/user-attachments/assets/30677a0f-dc1f-428f-ac4b-f8f98a67d5e1">  

*Figure 13: Occupation Change by Skills Gap*

- In Figure 13, the skills gap vs. the likelihood to change occupation doesn’t stand out significantly. This means that how well an individual's skills match their job requirements, on a 1-10 scale, doesn’t directly correlate to changing occupations.

<img width="550" alt="figure 14" src="https://github.com/user-attachments/assets/2f89cef1-7c91-4884-9751-464b551c87b0">

*Figure 14: Occupation Change by Family Influence*

- In Figure 14, no one significantly stands out when looking at family influence vs. the likelihood to change occupation. This means that the degree of influence an individual's family has on their career choice doesn’t directly correlate to changing occupations.

<img width="550" alt="figure 15" src="https://github.com/user-attachments/assets/b591c4d0-c9ed-4371-8378-66fb7137fd32">

*Figure 15: Occupation Change by Mentorship Available*

- In Figure 15, when looking at mentorship availability vs. the likelihood to change occupations, no one significantly stands out. This means that whether an individual has access to a mentor doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 16" src="https://github.com/user-attachments/assets/72aa1a54-3489-4b50-9445-576cd618c447">  

*Figure 16: Occupation Change by Certifications*

- In Figure 16, when looking at certification vs. the likelihood to change occupation, no one significantly stands out. This means that the number of certifications an individual has doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 17" src="https://github.com/user-attachments/assets/d096b147-eb77-4b01-af65-abe6cd9bb3d6">

*Figure 17: Occupation Change by Freelance Experience*

- In Figure 17, when looking at freelance experience vs. the likelihood to change occupation, no one significantly stands out. This means that whether an individual has freelanced doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 18" src="https://github.com/user-attachments/assets/405f9341-00f1-4671-a90f-7738f9cfdbf5">

*Figure 18: Occupation Change by Geographic Mobility*

- In Figure 18, there isn’t much of a difference when looking at geographic mobility vs. the likelihood to change occupation. This means whether an individual is willing to relocate doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 19" src="https://github.com/user-attachments/assets/6c9d80be-421a-4433-8ab6-6b1f4219a583">  

*Figure 19: Occupation Change by Professional Networks*

- In Figure 19, no one stands out significantly when looking at professional networks vs. the likelihood to change occupation. This means that the measure of how strong an individual's professional network is, on a 1-10 scale, doesn’t directly correlate to changing occupations.

<img width="550" alt="figure 20" src="https://github.com/user-attachments/assets/5a46f4f7-529b-42e4-907d-0c476019a7fb">

*Figure 20: Occupation Change by Career Change Events*

- In Figure 20, no one stands out significantly when looking at career change events vs. the likelihood to change occupation. This means that the number of career changes an individual has made doesn’t directly correlate to changing occupations.
  
<img width="550" alt="figure 21" src="https://github.com/user-attachments/assets/766319d0-bd3a-4f28-bb20-d0cbae1e09e9">

*Figure 21: Occupation Change by Technology Adoption*

- In Figure 21, when looking at technology adoption vs. the likelihood to change occupation, no one stands out significantly. This means that the measure of an individual's comfort level with adopting new technologies, on a 1-10 scale, doesn’t directly correlate to changing occupations.

To visualize all of the data together I decided to use a correlation matrix, to see if what I saw on the bar graphs was true.

<img width="712" alt="correlation matrix" src="https://github.com/user-attachments/assets/fc89f5c5-fd70-4598-8cdd-8f785783d1c6">

*Figure 22: Correlation Matrix*

- In Figure 22, the numbers are on a scale from -0.6 to 1.0, with 1.0 being a perfect positive correlation. You can see that the only three variables that correlate with the ‘Likely to Change Occupation’ variable are ‘Job Satisfaction,’ ‘Salary,’ and ‘Career Change Interest.’
  
## Modeling

To model the variables, I had to prepare the data. From the preprocessing steps, I noticed that the three variables that contributed the most were ‘Job Satisfaction,’ ‘Salary,’ and ‘Career Change Interest.’ To accommodate this, I made an ```X_data_all``` version that contains all independent variables, and an ```X_data``` version with all independent variables, excluding those three. Next, I split and scaled the data.

```python
#setting the data for x and y
X_data_all = data.drop(['Likely to Change Occupation'], axis=1) #all variables
X_data = data.drop(['Likely to Change Occupation','Job Satisfaction', 'Salary', 'Career Change Interest' ], axis=1)
y_data = data['Likely to Change Occupation'].values.reshape(-1, 1)

#Splitting data
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_data_all, y_data, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

#x and y scale
yscaler = StandardScaler()
y_train_scaled_all = yscaler.fit_transform(y_train_all).ravel()
y_test_scaled_all = yscaler.transform(y_test_all).ravel()
y_train_scaled = yscaler.fit_transform(y_train).ravel()
y_test_scaled = yscaler.transform(y_test).ravel()
```
Then, I ran the following code to help me find the best ```max_depth``` values for the decision tree models:
``` python
param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
clf = DecisionTreeClassifier ()

grid_search_all = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

grid_search_all. fit(X_train_all, y_train_all)
grid_search. fit(X_train, y_train)

best_max_depth_all = grid_search_all. best_params_['max_depth']
best_max_depth = grid_search. best_params_['max_depth']

print("Best max_depth_all:", best_max_depth_all)
print("Best max_depth:", best_max_depth)
```

I then created models for decision tree classifier, logistic regression, and random forest classifier. I created a decision tree classifier model because they are used to solve classification problems and can categorize variables based on their learning features. Next, I created a logistic regression model because it allows you to predict binary outcomes based on independent variables. Lastly, I created a random forest classifier model because it can handle larger datasets more efficiently and provides insight to feature importnaces.

```python
#decision tree classifier
tree_model_all = DecisionTreeClassifier(max_depth= best_max_depth_all,random_state=42)
tree_model_all.fit(X_train_all, y_train_all.ravel())
y_pred_tree_all = tree_model_all.predict(X_test_all)
rmse_tree_all = np.sqrt(np.mean((y_test_all - y_pred_tree_all)**2))
print("RMSE for Decision Tree (all variables):", rmse_tree_all)

tree_model = DecisionTreeClassifier(max_depth=best_max_depth, random_state=42)
tree_model.fit(X_train, y_train.ravel())  
y_pred_tree = tree_model.predict(X_test)
rmse_tree = np.sqrt(np.mean((y_test - y_pred_tree)**2))
print("RMSE for Decision Tree:", rmse_tree)
print()

#logistic regression
logistic_all = LogisticRegression(max_iter=2000,penalty='l2', C=1.0, solver='lbfgs',random_state=42)
logistic_all.fit(X_train_scaled_all, y_train_all.ravel())
y_pred_log_all = logistic_all.predict(X_test_scaled_all)
rmse_log_all = np.sqrt(np.mean((y_test_all - y_pred_log_all)**2))
print("RMSE for Logistic Regression (all variables):", rmse_log_all)

logistic = LogisticRegression(max_iter=2000,penalty='l2', C=1.0, solver='lbfgs',random_state=42)
logistic.fit(X_train_scaled, y_train.ravel())
y_pred_log = logistic.predict(X_test_scaled)
rmse_log = np.sqrt(np.mean((y_test - y_pred_log)**2))
print("RMSE for Logistic Regression:", rmse_log)
print()

#random forest classifier
forest_all = RandomForestClassifier(n_estimators=100, random_state=42)
forest_all.fit(X_train_all, y_train_all.ravel())
y_pred_forest_all = forest_all.predict(X_test_all)
rmse_forest_all = np.sqrt(np.mean((y_test_all - y_pred_forest_all)**2))
print("RMSE for Random Forest (all variables):", rmse_forest_all)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train.ravel())
y_pred_forest = forest.predict(X_test)
rmse_forest = np.sqrt(np.mean((y_test - y_pred_forest)**2))
print("RMSE for Random Forest:", rmse_forest)
```

For each model I printed out the root mean squared error. The closer the error it is to 0 the better because the difference between the predicted and actual values is small, meaning the model made good predictions. 

Below is a snippet of the printed out RMSE values:

<img width="552" alt="RMSE" src="https://github.com/user-attachments/assets/e671adff-0b45-4c9c-8bd2-6f7207353b45">


*Figure 23: Root Mean Squared Error Values*

- Figure 23 shows the printed-out RMSE values for each model created.

Then I graphed all three models together to see which models were better.
Here are the plots of the REC curves for both ```X_data_all``` and ```X_data```:

<img width="335" alt="all rec curves" src="https://github.com/user-attachments/assets/5b367962-1135-483e-a732-8668f91ef00b"> <img width="310" alt="limited REC curves" src="https://github.com/user-attachments/assets/404d5ad7-54fd-49c7-a634-c3e787015616">


*Figure 24: REC Curves (All Variables)* & *Figure 25: REC Curves*

- Figures 24 and 25 show all three REC curves, each in a different color for both ```X_data_all``` and ```X_data```. The lines in figure 24 are almost identical, while the random forest model in figure 25 is slightly off from the others.
  
Next, I made correlation matrices to see how well each model correctly predicts outcomes. In binary representation, the answer to the question, "Is this person likely to change their occupation?" is 0 for unlikely and 1 for likely.

Below are the matrices:

<img width="325" alt="lr all" src="https://github.com/user-attachments/assets/f8372b8e-7488-4636-8fe3-af49e68e6963"> <img width="325" alt="lr" src="https://github.com/user-attachments/assets/d8c24656-a44c-4466-ab42-09c94d20182e">

*Figure 26: Logistic Regression Confusion Matrix (All Variables)* & *Figure 27: Logistic Regression Confusion Matrix*

- Figures 26 and 27 show the confusion matrices for both logistic regression models. A gradient colors the number of actual and predicted values.
  
<img width="325" alt="tree all" src="https://github.com/user-attachments/assets/cffd2f7b-4d1e-4ecf-9c37-393e21fa1768"> <img width="350" alt="tree" src="https://github.com/user-attachments/assets/247f3271-9639-4587-96ac-771ae1329ca3">

*Figure 28: Decision Tree Confusion Matrix (All Variables)* & *Figure 29: Decision Tree Confusion Matrix*

- Figures 28 and 29 show the confusion matrices for both decision tree models. A gradient colors the number of actual and predicted values.
  
<img width="325" alt="forest all" src="https://github.com/user-attachments/assets/55961c15-c1e6-455c-ae5d-6ceb13aca5dd"> <img width="325" alt="forest" src="https://github.com/user-attachments/assets/276d8599-40f5-44f8-a030-a5b0482e9fa2">

*Figure 30: Random Forest Confusion Matrix (All Variables)* & *Figure 31: Random Forest Confusion Matrix*

- Figures 30 and 31 show the confusion matrices for both random forest models. The number of actual and predicted values is colored by a gradient.
  
Next, I created receiver-characteristic operator (ROC) curves for the models to show their performance across all thresholds. The model also shows the area under the curve (labeled AUC). The closer this number is to 1, the better the model performs and the better it can correctly classify outcomes. Below are the ROC Curves for all models:

<img width="325" alt="lr roc all" src="https://github.com/user-attachments/assets/8fe98081-f826-4dd1-8f87-dfa8031835fe"> <img width="325" alt="lr roc" src="https://github.com/user-attachments/assets/fd1ff5bc-dc05-4330-9357-0e107f616660">

*Figure 32: Logistic Regression ROC Curve (All Variables)* & *Figure 33: Logistic Regression ROC Curve*

- Figures 32 and 33 show the ROC curves for both logistic regression models. Notice the difference in graphs when only three variables were removed.
  
<img width="325" alt="tree roc all" src="https://github.com/user-attachments/assets/04ae62db-88ed-4b62-9b4d-82b19b1d5c71"> <img width="325" alt="tree roc" src="https://github.com/user-attachments/assets/80b6d0c2-6c6e-42fa-8a9a-12e63e2ee608">

*Figure 34: Decision Tree ROC Curve (All Variables)* & *Figure 35: Decision Tree ROC Curve*

- Figures 34 and 35 show the ROC curves for both decision tree models. Again, notice the difference between the two graphs.
  
<img width="325" alt="forest roc all" src="https://github.com/user-attachments/assets/6b9f7dbe-687b-42e7-9a5a-a778c33d7997"> <img width="325" alt="forest roc" src="https://github.com/user-attachments/assets/7425702b-303c-430e-a1c3-bb9954f515f8">

*Figure 36: Random Forest ROC Curve (All Variables)* & *Figure 37: Random Forest ROC Curve*

- Figures 36 and 37 show the ROC curves for both random forest models. Notice again the difference in the graphs.
  
Next, I graphed the decision trees to help visualize the flowchart used to predict outcomes. The tree contains the variables that can correctly predict whether a person is likely to change their occupation.

<img width="325" alt="tree graph all" src="https://github.com/user-attachments/assets/d077dde5-e79b-4ec4-bb02-479b4cb70897"> <img width="360" alt="tree graph" src="https://github.com/user-attachments/assets/31945812-0d23-4a35-b7fa-89d83c0e19c7">

*Figure 38: Decision Tree (All Variables)* & *Figure 39: Decision Tree*

- Figures 38 and 39 show the decision tree graphs made from the two decision trees. The best ```max_depth``` for each tree was calculated earlier and used for each tree.

Finally, I created used the random forest models to create feature importance graphs to show how much each variable contributes to outcome compared to each other.

<img width="345" alt="feature imprtance all" src="https://github.com/user-attachments/assets/89177eaf-7525-4a54-b3ee-bbb8d30db928"> <img width="360" alt="feature importance" src="https://github.com/user-attachments/assets/412f7898-b6bf-40e8-96bb-5dc7df3bb8bd">

*Figure 40: Feature Importances (All Variables)* & *Figure 41: Feature Importances*

- Figures 40 and 41 show the feature importance made from the random forest models. While Figure 41 shows more features with importance, Figure 40 shows features with higher importance values.
  
## Results

From the bar graphs (Figures 1-21) alone, you can predict that those who rated their job satisfaction as low, do not make a lot of money, and are interested in changing their career are more likely to change their occupation. The confusion matrix (Figure 22) proves this to be true. While the values along the diagonal are red, because it is a correlation of itself, we want to look at the variables that correlate to ‘Likely to Change Occupation.’ The first variable, ‘Job Satisfaction,’ indicates a -0.60 correlation to ‘Likely to Change Occupation.’ This isn’t a perfect correlation, but it generally means that lower values of job satisfaction correlate to “higher” values of likely to change occupation, and higher values of job satisfaction correlate to “lower” values of likely to change occupation. In this case, 1 is the highest value, corresponding to most likely, and 0 is the lowest value, corresponding to least likely. The next variable with some correlation is ‘Salary.’ With a -0.19 correlation, this isn’t as strong as the last variable, but it is still of some importance. The variable with a positive correlation is ‘Career Change Interest.’ With a value of 0.43, the correlation is almost as strong as the negative correlation that the ‘Job Satisfaction’ variable has. All other variables are less than 0.1, so the correlation might as well be 0.

The RMSE values in Figure 23 for all models are very close together. Typically, RMSE is not ideal for classification problems because it doesn’t consider whether the binary classes (0 or 1) are correct. When understanding RMSE values, typically, the lower the value, the better. 

Figure 24 shows the REC Curves when all independent variables are considered. Notice that all graphs do not start at the origin. This is because the models are for a classification problem. If a model can correctly classify a non-zero number of samples without any tolerance for error, the percentage of correct predictions is already above 0 when the absolute error is 0. All of the models were able to achieve this. Figure 25 is similar, but the random forest started at a lower percentage when the error was 0. The models achieved 100% correct predictions when the error reached one because of the binary classification 0 or 1). This is why when I defined the function to plot the curves, I set the max tolerance to 5 because I knew it would plateau at an error of 1.

The confusion matrices in Figure 26 - 31 show how well all the models performed. This matrix takes in all of the samples from the test set and counts the number of correct or incorrect outcomes the model predicted. The color gradient corresponds to the number of samples. The x-axis represents the predicted values from the linear regression model, and the y-axis represents the actual values from the dataset. The ‘unlikely’ is represented in binary by 0 for unlikely to change occupation, and ‘likely’ is represented in binary by 1 for likely to change occupation. This can be read as a 2x2 matrix. The top left corner represents the number of samples predicted to be unlikely and unlikely (true negatives). The top right corner represents the number of samples predicted to be likely and unlikely (false positives). The bottom left corner represents the number of samples predicted to be unlikely and likely (false negatives). The bottom right corner represents the number of samples predicted to be likely and likely (true positives). Figures 26, 28, and 30 show that the  models where all variables were used made more correct predictions than incorrect ones. Figures 27, 29, and 31 show that the models without the three variables predicted more positives (both true and false). This can be due to several things, including a model that is too simple, over-predicting the positive class, or having a low decision threshold.


The ROC curves in Figures 32 - 37 are graphical representations of the relationship between the true positive rate and the false positive rate. The area under the curve (AUC) measures this relationship. The orange line represents a completely random prediction that performs no better than simply guessing with an AUC of 0.5. A perfect model would have an AUC of 1. Figure 32 shows that the AUC of the logistic regression model using all variables is 0.97, which means the model has a 97% probability of correctly distinguishing between classes. Figure 33 has an AUC of 0.5, so without the three variables, it performed no better than a random guessing model. Using the decision tree model (Figure 34) and random forest model (Figure 36) with all variables, the AUC was 1.00, which means the model could make 100% correct predictions. Figure 35 has an AUC of 0.51, so it performed slightly better than random predictions but still poorly. Like the logistic regression model without the three variables, Figure 37 has an AUC of 0.5, so it did no better than random guessing. The models from the graphs in Figures 33, 35, and 37 performed poorly because, without the three variables, the other independent variables didn't have a strong relationship with the target variable.

The decision tree in Figure 38 was set with a ```max_depth``` 3, while Figure 39 was set to 2. This was found using the code mentioned earlier. These models take in the samples from the training set and split them into nodes that can lead to a correct classification. Each node contains the feature being split, number or samples, value, and class. The value refers to how many samples are in each class. For example, in the top node of Figure 38, the value is [3512, 4908], so there are 3512 samples of class unlikely and 4908 samples of class likely. In this project, unlikely and likely answer the question of how likely this individual is to change occupations. Then, the class displayed is whichever class has more samples in that node. As expected, Figure 38 split the variables 'Job Satisfaction,' 'Career Change Interest,' and 'Salary.' Figure 39 used the variables 'Job Opportunities' and 'Current Occupation.' Figure 38 could make correct predictions by the end, while Figure 39 could not. The ```max_depth``` would have to be way higher, but that isn't ideal. The code said it should be set to 2 because, looking at the variables, none have a strong enough relationship with the target variable to make a decent tree.

Lastly, Figures 40 and 41 used the random forest models to show the feature importance of each variable. Figure 40 confirms that the only three variables that are significantly important in predicting whether an individual is likely to change their occupation are ‘Job Satisfaction,’ ‘Career Change Interest,’ and ‘Salary.’ When taking those variables out, Figure 41 shows that multiple variables have importance but very little according to the y-axis values. These are all less than 0.14, which is very little compared to the values in Figure 40. This confirms all other graphs because, with very small importance, models will find it harder to find a correlation to the target variable and end up making random predictions.


## Conclusion

In conclusion, I wanted to find relationships between variables and use machine learning tools and techniques to correctly predict whether an individual is likely to change their occupation. After several different graphs and models were made, the following conclusions were made.

With all variables included, the logistic regression, decision tree, and random forest models achieved high accuracy with a very small margin of error. The models without the three variables proved poor due to the limited relationship between the features and the target variable. This shows that when making a model where a few features are heavily relied on, the other features won't matter. In this case, when looking at the models where all features had similar importance (models without the three), it was very hard to predict the outcome with the remaining features correctly.

Something that could have been improved was using the entire dataset to get a better representation of all ages. If I had not restricted the ages, maybe more than three variables could have contributed to making correct predictions. Either way, I have found that for people ages 20-30, job satisfaction, career change interest, and salary are the variables that can be used to predict whether an individual is likely to change their occupation.

The conclusions from this project can be applied to future projects. It can be expanded by knowing that logistic regression, decision tree, and random forest models performed well in this classification problem. A machine with a higher capacity could make predictions with larger datasets and take in more variables. Outliers must be filtered out with larger datasets so the data isn’t skewed. Some variables that can be added are marriage status, geographical location, number of children, or stress levels.


## References
[1] Paliwal, Jahnavi. 2024. “Field Of Study vs Occupation.” Kaggle. https://www.kaggle.com/datasets/jahnavipaliwal/field-of-study-vs-occupation/data.



