**Atmospheric and Oceanic Science C111 Final Project** 

**Dr. Alexander Lozinski**

**December 6, 2024**

For this project, I used machine learning techniques to analyze what variables contribute to University student's sleep patterns.

## Introduction 

  Many people know the saying, “College is the best years of your life.” While college is important for building relationships and figuring out what you want to do with your life, sleep plays an important role. Sleep is important for college students because it heavily contributes to their overall health and well-being. Many things contribute to how much sleep a student gets. In this report, you will see the different variables that I looked at to try and figure out if there was one that affected sleep the most.
	By using the “Student Sleep Patterns” dataset on Kaggle, I chose variables that I believed would contribute the most to the amount of sleep. I modeled these variables with a plot to get a visual representation of the data. I then used linear regression to determine the strength of the relationship between each variable. 



## Data

Here is an overview of the dataset, how it was obtained and the preprocessing steps taken, with some plots!

![](assets/IMG/datapenguin.png){: width="500" }

*Figure 1: Here is a caption for my diagram. This one shows a pengiun [1].*

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


