Baseball Salary Prediction
---
It's written for Python 2.7.2.

We tried to predict Baseball playes salaries.

Assumption: that salary will be based on player's performance.

The website called [Statiz](http://www.statiz.co.kr/) has players' performance data for season.
The data we used to analyze was crawled on there.

There are many features to predict players' salary, in addition some features' correlation are too high to use general linear regression

Because of that, PCA regression analsis was choosed to make possible to use linear regression although some of them have high correlation.

**Baseball_Salary_Prediction.ipynb** has more detail about salary prediction method.

I am so appreciate if you try to read that file.
