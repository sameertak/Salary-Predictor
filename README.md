# Salary-Predictor
Made a Predictive Regressor Model which can predict the salary of a Software Developer.

You can find the working Web [here](https://share.streamlit.io/sameertak/salary-predictor/app.py).

The Web App has two pages:-
1) Prediction
2) Explore

Both can be accessed from the top left "[>]" option

What's in the code?
----------------------------------
First of all, I have downloaded the csv file from [Stack Overflow](https://insights.stackoverflow.com/survey)
Which contains the Survey Dataset of Software Engineers Salaries with their Education Levels and all other required things.
I've cleaned and transformed the data as per the need.
And applied models, in which Decision Tree gave the best results. (i.e. with least loss)
So, I saved the model with the help of 'pickle'.

Then, I used [StreamLit](https://streamlit.io/) for deploying my model on the web.
StreamLit is an easy to use Web Deployment Application in Python. I recommend to use it if you haven't tried it yet.

The code used is basic and easy to understand.

How to run the file?
-----------------------------
Just download or clone the files in your system and then from command prompt (or Anaconda Powershell) head towards the root directory.
And run the following code:-
```
conda activate ml
```
```
streamlit run app.py
```

Before running the file you need to make sure you have all the dependencies which are required.
(ml --> because here I've made a new virtual environment. You can skip the first line of code.)

What are the dependencies?
----------------------------------
The requirements list is included in the root folder, which contains all the necessary libraries required for the project.
Use the following code to install automatically the required libraries.
```
pip install -r /path/to/requirements.txt
```
