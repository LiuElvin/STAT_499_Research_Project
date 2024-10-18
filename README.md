# Classify High-Dimensional Data

# Project Overview

In machine learning, classification is a task that assigns a class label to examples from the problem domain. However, high dimensionality poses significant statistical challenges and renders many traditional classification algorithms impractical to use.

In this project, we will first learn some classical supervised classification techniques (decision trees, random forests, support vector machines, etc.) and discuss the curse of dimensionality.

Next, we will mainly explore Penalized Discriminant Analysis (PDA), which is designed to classify high-dimensional data as an extension of the classical Linear Discriminant Analysis. It classifies data by finding the optimal lower-dimension projections that reveal “interesting structures” in the original dataset.

Finally, we will implement PDA to analyze a real-life dataset of the student’s choice and some simple toy examples.

# Stuff

1. In machine learning, classification is a task that assigns a class label to examples from the problem domain. However, high dimensionality poses significant statistical challenges and renders many traditional classification algorithms impractical to use.

2. In this project, I learnt some classical supervised classification techniques and discussed the curse of dimensionality. After that, I explored Penalized Discriminant Analysis (PDA), which is designed to classify high-dimensional data as an extension of the classical Linear Discriminant Analysis. It classifies data by finding the optimal lower-dimension projections that reveal “interesting structures” in the original dataset.

3. Afterwards, I implemented PDA to analyze a real-life colon cancer dataset alongside a simple toy example with large dimension count but small sample size.

4. The results were summarized in my end of quarter presentation. It focussed primarily on the curse of dimensionality, LDA, PDA, their applications on both a toy and real-life dataset. The full slides and presentation can be found in the link [here](https://docs.google.com/presentation/d/1oSrP5NRSWhoQRwXVjOmNRvVN8d4FA-LlYU0DCK1PzhE/edit?usp=sharing).

# Files

- `./data` contains the colon cancer data used for projections.
- `./code` includes all of the code.
  - `./code/annotation`: annotation.

---

# Contributions

- The annotation file is merely an annotation of a pre-existing package by Eun-Kyung Lee. The link to her code is provided [here](https://github.com/EK-Lee/classPP/blob/master/R/PPindex.R).

- Zhaoxing Wu, my supervisor, provided the cirriculum and a lot of guidance for the direction of my project.

# Learning Outcomes

- Learnt theoretics behind classical supervised classification techniques.

- Learnt the curse of dimensionality and how it ruins traditional machine learning models.

- Learnt the concept of Penalized Discriminant Analysis as an extension of Linear Discriminant Analysis.
