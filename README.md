**Atmospheric and Oceanic Science C111 Final Project** 

**Dr. Alexander Lozinski**

**December 6, 2024**

For this project, I used machine learning techniques to analyze what variables contribute to University student's sleep patterns.

## Introduction 

Many people know the saying, “College is the best years of your life.” While college is important for building relationships and figuring out what you want to do with your life, sleep plays an important role in this process. Sleep is important for college students because it heavily contributes to their overall health and well-being. Many things contribute to how much sleep a student gets. In this report, you will see the different variables that I looked at to try and figure out if there was one that affected sleep the most.

By using the “Student Sleep Patterns” dataset on Kaggle, I chose variables that I believed would contribute the most to the amount of sleep. I modeled these variables with a plot to get a visual representation of the data. I then used linear regression to determine the strength of the relationship between each variable. 



## Data
[Click here to view the dataset I used!](https://www.kaggle.com/datasets/arsalanjamal002/student-sleep-patterns/data)

The dataset includes 500 rows, each representing a different student. With 14 columns, each is a different attribute contributing to sleep-related information. These 14 columns include student ID, age, gender, year in school, total hours of sleep per night, average study time, average screen time, average caffeine intake, average exercise time, sleep quality, weekday and weekend sleep start, and weekend and weekday sleep end. The website details each one and how the numbers are inputted into the dataset file. For this project, I decided to use total hours of sleep as my dependent variable because I want to see what independent variables affect the amount of sleep.



## Preprocessing Steps
## Modelling

Here are some more details about the machine learning approach, and why this was deemed appropriate for the dataset. 

The model might involve optimizing some quantity. You can include snippets of code if it is helpful to explain things.

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
clf.predict([[0, 0, 0, 0]])
```

This is how the method was developed.

## Results

Figure X shows... [description of Figure X].

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


