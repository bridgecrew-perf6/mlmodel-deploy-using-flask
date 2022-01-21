This project is going to demonstrate how to take a trained ML model as a REST endpoint using Flask.

The dataset comes from [Kaggle.com](https://www.kaggle.com/uciml/adult-census-income). This data was extracted from the 1994 Census bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics). A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0)). The prediction task is to determine whether a person makes over $50K a year.

There are 32561 records with 15 features.

We are going to use this dataset to train a classification model that will predict whether a given individual will have income above $50K per year and deploy it as inside a Flask web app. 
