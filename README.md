# Classify High-Dimensional Data

# General Project Overview

In machine learning, classification is a task that assigns a class label to examples from the problem domain. However, high dimensionality poses significant statistical challenges and renders many traditional classification algorithms impractical to use.

In this project, we will first learn some classical supervised classification techniques (decision trees, random forests, support vector machines, etc.) and discuss the curse of dimensionality.

Next, we will mainly explore Penalized Discriminant Analysis (PDA), which is designed to classify high-dimensional data as an extension of the classical Linear Discriminant Analysis. It classifies data by finding the optimal lower-dimension projections that reveal “interesting structures” in the original dataset.

Finally, we will implement PDA to analyze a real-life colon cancer dataset alongside some simple toy examples. Comparisons are also drawn to the performance of each model.

A link to the full page with all of the SPA DRP projects of Autumn 2023 (including mine) is provided [here](https://spa-drp.github.io///past-projects/2023-autumn/).

My results were summarized in an end of quarter presentation. The full slide deck can be found [here](https://docs.google.com/presentation/d/1oSrP5NRSWhoQRwXVjOmNRvVN8d4FA-LlYU0DCK1PzhE/edit?usp=sharing).

# Files

- `./data` contains the colon cancer dataset.
- `./code` includes all of the code.
  - `./code/annotation`: annotations of useful R functions.

# Results

Below are the 1-D and 2-D projections of a dummy simulation.

I've also provided some example code if reproduction is of interest.

Set up code:

```{r}
library(devtools)
install_github("EK-LEE/classPP")
library(classPP)
```

---

# Contributions

- The annotation file is merely my annotations over a pre-existing package by Eun-Kyung Lee. The link to her code is provided [here](https://github.com/EK-Lee/classPP/blob/master/R/PPindex.R).

- Zhaoxing Wu, my mentor, provided a list of machine learning literature for my understanding along with a lot of guidance for the direction of my project.

  - A link to her paper, which much of my project is based off of, can be found [here](https://jds-online.org/journal/JDS/article/1326/info).
 
- The dataset was fetched from Notterman, et al, Cancer Research vol. 61: 2001.

  - The link to the raw files can be accessed via [Princeton](http://genomics-pubs.princeton.edu/oncology/database.html).

# Learning Outcomes

- Learnt theoretics behind classical supervised classification techniques.

- Learnt the curse of dimensionality and how high dimension low sample size data ruins traditional machine learning models.

- Learnt the concept of Penalized Discriminant Analysis as an extension of Linear Discriminant Analysis.

- Created and ran multiple projection functions on real training / testing data and created corresponding graphs in R.

- Learnt how to create ideal dummy simulations and how to generalize results off it.
