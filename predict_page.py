import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

reg = data['model']
le_country = data['le_country']
le_education = data['le_education']


def show_predict_page():
    st.title("Software Developer Salary Predictor")
    st.write("""### We need some information to predict the salary.""")

    countries = (
        'United States of America',
        'India',
        'Germany',
        'United Kingdom of Great Britain and Northern Ireland',
        'Ireland',
        'Canada',
        'France',
        'Brazil',
        'Spain',
        'Netherlands',
        'Australia',
        'Poland',
        'Italy',
        'Russian Federation',
        'Sweden',
        'Turkey',
        'Switzerland',
        'Israel',
        'Norway'
    )

    educations = (
        'Bachelor’s Degree',
        'Master’s Degree',
        'Post Graduate',
        'Less then Bachelors'
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", educations)

    experience = st.slider("Years of Experience", 0,50,3)
    click = st.button('Calculate Salary')

    if click:
        X =np.array([[country, education, experience]])
        X[:,0] = le_country.transform(X[:,0])
        X[:,1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = reg.predict(X)

        st.subheader(f'The Estimated Salary is ${salary[0]:.2f}')