
<img width="1395" alt="cover" src="https://github.com/user-attachments/assets/886add33-86f3-4481-8a91-510b7822cae5">


**A Report on Career Change Predictions**

**Atmospheric and Oceanic Science C111 Final Project** 

**Dr. Alexander Lozinski**

**December 6, 2024**

For this project, I used machine learning techniques to analyze the factors affecting whether an individual wants to change their career.
## Introduction 

Many people go to college, get a degree in a field they're interested in, and move on to get a job unrelated to their field of study. The question arises: does your field of study determine your career path? In this report, you will see the different variables that could lead to someone wanting to change their occupation.

Using the “Field of Study vs Occupation” dataset on Kaggle, I looked at variables that I believed would contribute to an individual, ages 20-30, wanting to change their occupation. I decided to use supervised learning because I wanted to make predictions based on data patterns between the relationships of variables. I modeled these variables with data plots to get a visual representation of the data. I made a heat correlation map to find the variables that contribute most to wanting to change their career. I then made REC curves for lasso regression, SVR, decision tree, and logistic regression to show how well different models can predict these values. I then made a confusion matrix to visually show how well these models can predict the data. I also made a logistic regression model and showed the ROC curve to go with it. I made a decision tree plot to follow and show what values within variables lead to whether an individual is likely to change their occupation or not. Lastly, to confirm the correct variable was being used, I plotted the feature importance to clearly show which variables contribute the most to these predictions.




## Data
[Click here to view the dataset I used!](https://www.kaggle.com/datasets/jahnavipaliwal/field-of-study-vs-occupation/data)

The dataset includes 38,444 rows, each representing a different person. I narrowed it down to ages 20-30 (10,525 rows) to see how it affects people closer to me. With 22 columns, each attribute contributes to different information about each person. These 22 columns include their field of study, current occupation, age, gender, years of experience, education level, industry growth rate, job satisfaction, work-life balance, job opportunities, salary, job security, career change interest, skills gap, family influence, mentorship available, certifications, freelancing experience, geographic mobility, professional networks, career change events, technology adoption, and likely to change occupation. The website details each one and how the numbers are inputted into the dataset file. For this project, I used likely to change occupation as my dependent variable to see what independent variables affect that likelihood, and I used them for predictions.



## Preprocessing Steps

The first thing I had to do was upload the dataset into Google Colab. Next, I noticed the data set was really big so I narrowed it down to ages 20-30 (10,525 rows) to get a better representation of how it affects people closer in age to me. I then checked the data types to make sure I could use them.  After that, I had to convert the variables that were strings into integers so the data could be graphed. These variables included gender, family influence, field of study, occupation, education level, and industry growth rate. Using data.head() I was able to check that all the variables were integers and then proceeded to use data.describe() to look at averages.


Below is a snippet of the code where I cleaned up the data.

<img width="941" alt="cleaning" src="https://github.com/user-attachments/assets/e3d66c80-2056-4ae1-9a6b-6a93cb9ef298"> 


Below are the 5 rows that were printed out with data.head(5) that show all variables with integer values.

<img width="1261" alt="Screenshot 2024-12-03 at 7 22 32 PM" src="https://github.com/user-attachments/assets/a69a7c6e-0d8e-4aa8-989f-24e8f9795051">

<img width="721" alt="Screenshot 2024-12-03 at 7 23 34 PM" src="https://github.com/user-attachments/assets/fc0af86f-0088-4e75-ba02-e41b9772bea3">


Using data.describe(), I was able to see averages of every variable; they are shown below.

<img width="1235" alt="Screenshot 2024-12-03 at 7 24 47 PM" src="https://github.com/user-attachments/assets/3febe9d6-c0bf-4cc0-8e7a-cc2fd2558e1c">

<img width="1182" alt="Screenshot 2024-12-03 at 7 26 16 PM" src="https://github.com/user-attachments/assets/c96a4736-943a-456b-aedb-5d37ff65e651">


Next, I made bar graphs for each variable with ‘Likely to Change Occupation’ as the dependent variable. The graphs are shown below:


<img width="566" alt="figure 1" src="https://github.com/user-attachments/assets/48ba93b5-fa68-4772-ac69-ef349a555dd9">

*Figure 1: Occupation Change by Field of Study*

<img width="566" alt="figure 2" src="https://github.com/user-attachments/assets/10adbf45-904c-4c26-b6fb-07ead31b21cc">

*Figure 2: Occupation Change by Current Occupation*

<img width="566" alt="figure 3" src="https://github.com/user-attachments/assets/12d7ebab-0952-481d-8b4a-a5ae3e8dceaa">

*Figure 3: Occupation Change by Age*

<img width="566" alt="figure 4" src="https://github.com/user-attachments/assets/050840dd-4d52-4974-843a-ba6fa3b0fa7c">

*Figure 4: Occupation Change by Gender*

<img width="567" alt="figure 5" src="https://github.com/user-attachments/assets/4a5919ed-31f7-4b6e-a1ec-7ede7fe5ddf3">

*Figure 5: Occupation Change by Education Level*

<img width="561" alt="figure 6" src="https://github.com/user-attachments/assets/57eb8c4d-d504-47bd-a0d1-c1451e531017">

*Figure 6: Occupation Change by Industry Growth Rate*

<img width="566" alt="figure 7" src="https://github.com/user-attachments/assets/c93f087c-72b2-45f7-a4cf-f31e6a00a76f">

*Figure 7: Occupation Change by Job Satisfaction*

<img width="567" alt="figure 8" src="https://github.com/user-attachments/assets/6298dcf0-2653-46da-9f1d-e3ed86ba4323">

*Figure 8: Occupation Change by Work-Life Balance*

<img width="566" alt="figure 9" src="https://github.com/user-attachments/assets/374dcefa-d847-4fc9-b406-fd2cebd7de36">

*Figure 9: Occupation Change by Job Opportunities*

<img width="565" alt="figure 10" src="https://github.com/user-attachments/assets/6b6e5296-f101-4979-b839-f02b61223e7b">

*Figure 10: Occupation Change by Salary*

<img width="566" alt="figure 11" src="https://github.com/user-attachments/assets/e7858780-aff9-4f70-980f-e65f96bcadc7">

*Figure 11: Occupation Change by Job Security*

<img width="567" alt="figure 12" src="https://github.com/user-attachments/assets/98bc2f7d-dcb2-45e0-a434-8b29ec6c938f">

*Figure 12: Occupation Change by Career Change Interest*

<img width="564" alt="figure 13" src="https://github.com/user-attachments/assets/30677a0f-dc1f-428f-ac4b-f8f98a67d5e1">

*Figure 13: Occupation Change by Skills Gap*

<img width="566" alt="figure 14" src="https://github.com/user-attachments/assets/2f89cef1-7c91-4884-9751-464b551c87b0">

*Figure 14: Occupation Change by Family Influence*

<img width="563" alt="figure 15" src="https://github.com/user-attachments/assets/b591c4d0-c9ed-4371-8378-66fb7137fd32">

*Figure 15: Occupation Change by Mentorship Available*

<img width="566" alt="figure 16" src="https://github.com/user-attachments/assets/72aa1a54-3489-4b50-9445-576cd618c447">

*Figure 16: Occupation Change by Certifications*

<img width="566" alt="figure 17" src="https://github.com/user-attachments/assets/d096b147-eb77-4b01-af65-abe6cd9bb3d6">

*Figure 17: Occupation Change by Freelance Experience*

<img width="566" alt="figure 18" src="https://github.com/user-attachments/assets/405f9341-00f1-4671-a90f-7738f9cfdbf5">

*Figure 18: Occupation Change by Geographic Mobility*

<img width="566" alt="figure 19" src="https://github.com/user-attachments/assets/6c9d80be-421a-4433-8ab6-6b1f4219a583">

*Figure 19: Occupation Change by Professional Networks*

<img width="564" alt="figure 20" src="https://github.com/user-attachments/assets/5a46f4f7-529b-42e4-907d-0c476019a7fb">

*Figure 20: Occupation Change by Career Change Events*

<img width="565" alt="figure 21" src="https://github.com/user-attachments/assets/766319d0-bd3a-4f28-bb20-d0cbae1e09e9">

*Figure 21: Occupation Change by Technology Adoption*

To visualize all of the data together I decided to use a correlation matrix, to see if what I saw on the bar graphs was true.

<img width="712" alt="correlation matrix" src="https://github.com/user-attachments/assets/fc89f5c5-fd70-4598-8cdd-8f785783d1c6">

*Figure 22*

## Modeling

To model the variables I had to prepare the data. I noticed from the preprocessing steps that the 3 variables that contributed the most were ‘Job Satisfaction’, ‘Salary’, and ‘Career Change Interest’ so I created 2 versions of the X_data. One version was with all variables used, and the other version was with just those 3 variables. Next, I split and scaled the data.

<img width="1151" alt="prepare data" src="https://github.com/user-attachments/assets/a4f6a7bf-bc38-4471-bddf-38067b562ec3">

*Figure 23*

I then created models for lasso regression, SVR, decision tree, and logistic regression. I created a lasso regression model it selects the most important features, making the predictions more correct. I created an SVR model because I wanted to minimize the error between the predicted values and actual values. I created a decision tree model to help visualize how the predictions were being made. Finally, I created a logistic regression model because this is a prediction problem and I wanted to find relationships between variables to make the correct predictions. For each model I printed out the root mean squared error. The closer the error it is to 0 the better because the difference between the predicted and actual values is small, meaning the model made good predictions. Then I graphed all 4 models together to see which models were better.


Below is a snippet of the printed out RMSE values:

<img width="424" alt="rmse" src="https://github.com/user-attachments/assets/c5d70a02-d3c8-46d8-8963-ae4ef0346f89">

*Figure 24: Root Mean Squared Error Values*

Here is the plot of the REC curves:

<img width="696" alt="curves" src="https://github.com/user-attachments/assets/fdf3ceb5-60c3-4c11-aa3a-a8997443fd0c">

*Figure 25: REC Curves*

Next, I used the logistic regression model to create a correlation matrix to see how well the model predicts outcomes correctly I chose to use the logistic model because logistic regression is a classification technique that is used to predict binary outcomes. In this case it is 0 for no and 1 for yes when answering the question, Is this person likely to change their occupation.

Below is the matrix:

<img width="656" alt="confusion matrix" src="https://github.com/user-attachments/assets/c63ae60a-e79b-4c47-ad8c-1f6bd087b659">

*Figure 26: Confusion Matrix*

Next I created a ROC curve with the logistic regression model to show how well the model performed. The model also shows the area under the curve (labeled AUC). Teh closer this number is to 1 means that the model has an excellent performance and a high ability to correctly classify outcome.

<img width="445" alt="ROC" src="https://github.com/user-attachments/assets/4c48a65a-5ac3-440d-bf5c-36d2f4613d14">

*Figure 27: ROC Curve*

Next, I created a decision tree to help visualize the flowchart that was used to predict outcomes. The tree contains the variables that can correctly predict whether a person is likely to change their occupation.

<img width="573" alt="decision tree" src="https://github.com/user-attachments/assets/ecb7ed67-8ba2-4ef4-9539-0d3384e53cbd">

*Figure 28: Decision Tree*


Finally, I created a feature importance graph to show how much each variable contributes to outcome compared to each other.

<img width="626" alt="features" src="https://github.com/user-attachments/assets/4425d17c-bf3b-45ec-9c93-50c8ec2ac444">

*Figure 29: Feature Importances*


Results From Preprossesing steps:
-------------------------
- In Figure 1, when looking at the fields of study modeled vs. the likelihood to change occupations, no one significantly stands out. This means that an individual's field of study doesn’t directly correlate to changing occupations.


- In Figure 2, when looking at current occupation vs. likelihood to change occupation, no one significantly stands out. This means that an individual's current occupation doesn’t directly correlate to changing occupations.


- In Figure 3, when looking at age vs. the likelihood to change occupation, no one significantly stands out. This means that an individual's age doesn’t directly correlate to changing occupations.


- In Figure 4, when looking at gender vs. the likelihood to change occupation, no gender significantly stands out. This means that an individual's gender doesn’t directly correlate to changing occupations.


- In Figure 5, when looking at education level vs. the likelihood to change occupation, no one significantly stands out. This means that an individual's education level doesn’t directly correlate to changing occupations.


- In Figure 6, individuals in an industry with a higher growth rate are more likely to change their occupation. Although there is a slight correlation, the difference between low, medium, and high isn't significant enough to differentiate between them.


- In Figure 7, individuals who rated their job satisfaction as four or lower on a 1-10 scale are almost certain to change their job satisfaction. This makes sense because if you aren’t enjoying your job, you want to find a different one.


- In Figure 8, no one stands out significantly when comparing work-life balance to the likelihood of changing occupations. The likelihood of changing occupations fluctuates, so an individual's rating of work-life balance on a 1-10 scale doesn’t directly correlate to changing occupations.


- In Figure 9, when looking at job opportunities vs. the likelihood to change occupation, no one significantly stands out. This means that an individual's number of job opportunities doesn’t directly correlate to changing occupations.


- Figure 10 shows a graph showing that individuals who make between $30k and $60k are much more likely to change occupations than someone who makes more than that.


- In Figure 11, no one stands out significantly when looking at job security vs. the likelihood to change occupation. This means that an individual's rating of job security, on a 1-10 scale, doesn’t directly correlate to changing occupations.


- In Figure 12, as one can expect, those interested in changing their career are more likely to change occupations.


- In Figure 13, the skills gap vs. the likelihood to change occupation doesn’t stand out significantly. This means that how well an individual's skills match their job requirements, on a 1-10 scale, doesn’t directly correlate to changing occupations.


- In Figure 14, no one significantly stands out when looking at family influence vs. the likelihood to change occupation. This means that the degree of influence an individual's family has on their career choice doesn’t directly correlate to changing occupations.


- In Figure 15, when looking at mentorship availability vs. the likelihood to change occupations, no one significantly stands out. This means that whether an individual has access to a mentor doesn’t directly correlate to changing occupations.


- In Figure 16, when looking at certification vs. the likelihood to change occupation, no one significantly stands out. This means that the number of certifications an individual has doesn’t directly correlate to changing occupations.


- In Figure 17, when looking at freelance experience vs. the likelihood to change occupation, no one significantly stands out. This means that whether an individual has freelanced doesn’t directly correlate to changing occupations.


- In Figure 18, there isn’t much of a difference when looking at geographic mobility vs. the likelihood to change occupation. This means whether an individual is willing to relocate doesn’t directly correlate to changing occupations.


- In Figure 19, no one stands out significantly when looking at professional networks vs. the likelihood to change occupation. This means that the measure of how strong an individual's professional network is, on a 1-10 scale, doesn’t directly correlate to changing occupations.


- In Figure 20, no one stands out significantly when looking at career change events vs. the likelihood to change occupation. This means that the number of career changes an individual has made doesn’t directly correlate to changing occupations.


- In Figure 21, when looking at technology adoption vs. the likelihood to change occupation, no one stands out significantly. This means that the measure of an individual's comfort level with adopting new technologies, on a 1-10 scale, doesn’t directly correlate to changing occupations.

- In Figure 22, the numbers are on a scale from -0.6 to 1.0, with 1.0 being a perfect positive correlation. You can see that the only three variables that correlate with the ‘Likely to Change Occupation’ variable are ‘Job Satisfaction,’ ‘Salary,’ and ‘Career Change Interest.’ 

Results From Modeling:
-------------

- Figure 23 shows the code I created before I could make the models. The X-data has two options, one with all variables and the other with only the most important three. I split the data and then used StandardScaler( ) because some models need the data to be scaled.

- Figure 24 shows the printed-out RMSE values for each model created.

- Figure 25 shows all four REC curves, each in a different color. Notice how two of the models start at 50 while the other two start at 0.

- Figure 26 shows the confusion matrix where the number of actual and predicted values are color coded.

- Figure 27 shows the ROC curve printed out in blue and the y=x line printed out as a reference, The area under the curve (AUC) is also printed.

- Figure 28 shows the decision tree the model made. It also displays the number of samples and squared error at each node to show how the model made these predictions.

- Figure 29 shows the importance of each feature (variable) in the dataset.




## Discussion

From Figure X, one can see that... [interpretation of Figure X].

## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)


