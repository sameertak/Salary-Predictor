import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('new_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

reg = data['model']
le_country = data['le_country']
le_education = data['le_education']
le_main = data['le_main']
le_employ = data['le_employ']
le_sex = data['le_sex']


def show_predict_page():
    st.title("Software Developer Salary Predictor")
    st.write("""### We need some information to predict the salary.""")

    mains = (
        'I am a developer by profession',
        'I am a student who is learning to code',
        'I am not primarily a developer, but I write code sometimes as part of my work',
        'I code primarily as a hobby',
        'I used to be a developer by profession, but no longer am',
        'None of these'
    )
    countries = (
        'Slovakia', 'Netherlands', 'Russian Federation', 'Austria',
        'United Kingdom of Great Britain and Northern Ireland',
        'United States of America', 'Malaysia', 'India', 'Sweden', 'Spain',
        'Germany', 'Peru', 'Turkey', 'Canada', 'Singapore', 'Brazil',
        'France', 'Switzerland', 'Malawi', 'Israel', 'Poland', 'Ukraine',
        'Viet Nam', 'Portugal', 'Italy', 'Bulgaria', 'Greece',
        'Iran, Islamic Republic of...', 'Ireland', 'Georgia', 'Uzbekistan',
        'Hungary', 'Belgium', 'Pakistan', 'Nigeria', 'Albania',
        'Bangladesh', 'Romania', 'Sri Lanka', 'Lithuania', 'Slovenia',
        'Croatia', 'Czech Republic', 'Denmark', 'Armenia', 'Lebanon',
        'Bahrain', 'Egypt', 'Nepal', 'Colombia', 'Indonesia', 'Australia',
        'Turkmenistan', 'Morocco', 'Chile', 'Serbia', 'New Zealand',
        'Estonia', 'Tunisia', 'Finland', 'Hong Kong (S.A.R.)',
        'United Arab Emirates', 'Argentina', 'Azerbaijan', 'Philippines',
        'Costa Rica', 'South Africa', 'Kosovo', 'Japan',
        'United Republic of Tanzania', 'Bolivia', 'Bosnia and Herzegovina',
        'Uruguay', 'South Korea', 'China', 'Norway', 'Belarus',
        'Luxembourg', 'Malta', 'Ethiopia', 'Madagascar', 'Kenya',
        'The former Yugoslav Republic of Macedonia', 'Botswana', 'Algeria',
        'Senegal', 'Mexico', 'Cyprus',
        'Venezuela, Bolivarian Republic of...', 'Cameroon', 'Jordan',
        'Dominican Republic', 'Ecuador', 'Syrian Arab Republic', 'Zambia',
        'Taiwan', 'Nomadic', 'Latvia', 'Guatemala', 'Paraguay', 'Iceland',
        'Haiti', 'Republic of Moldova', 'Kazakhstan',
        'Libyan Arab Jamahiriya', 'Afghanistan', 'Panama', "Côte d'Ivoire",
        'Cuba', 'Myanmar', 'Tajikistan',
        "Lao People's Democratic Republic", 'Yemen', 'Thailand', 'Qatar',
        'Democratic Republic of the Congo', 'Iraq', 'Mozambique',
        'Somalia', 'Andorra', 'Kyrgyzstan', 'Kuwait', 'Saudi Arabia',
        'Mauritania', 'Honduras', 'Angola', 'Oman', 'Swaziland', 'Sudan',
        'Guyana', 'Chad', 'El Salvador', 'Benin', 'North Korea',
        'Nicaragua', 'Dominica', 'Trinidad and Tobago', 'Ghana',
        'Barbados', 'Burundi', 'Micronesia, Federated States of...',
        'Zimbabwe', 'Mauritius', 'Gambia', 'Bahamas',
        'Congo, Republic of the...', 'Suriname', 'Djibouti',
        'Republic of Korea', 'Bhutan', 'Cambodia', 'Uganda', 'Rwanda',
        'Montenegro', 'Maldives', 'Saint Kitts and Nevis', 'Monaco',
        'Togo', 'Isle of Man', 'Jamaica', 'Belize', 'Palestine',
        'Mongolia', 'Burkina Faso', 'Liechtenstein', 'Saint Lucia',
        'Cape Verde', 'Brunei Darussalam', 'Namibia',
        'Central African Republic', 'Lesotho', 'Guinea', 'Liberia', 'Fiji',
        'Niger', 'Sierra Leone', 'San Marino',
        'Saint Vincent and the Grenadines', 'Tuvalu', 'Papua New Guinea',
        'Mali'
    )

    educations = (
        'Bachelor’s Degree',
        'Master’s Degree',
        'Post Graduate',
        'Less then Bachelors'
    )

    employments = (
        'Freelancer', 'Full-Time Student', 'Full-Time Employed',
        'Part-Time Student', 'Part-Time Employed',
        'Not Employed but looking for work', 'Retired'
    )

    sexs = (
        'Man', 'Prefer not to say', 'Woman',
        'Non-binary, genderqueer, or gender non-conforming'
    )
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", educations)
    employment = st.selectbox("Employment", employments)
    sex = st.selectbox("Gender", sexs)
    age = st.slider("Age", 15,60,10)
    main = st.selectbox("Main Branch", mains)
    experience = st.slider("Years of Coding", 1,50,5)

    click = st.button('Calculate Salary')

    if click:
        X = np.array([[main, employment, country, education, experience, sex, int(age)]])
        X[:, 0] = le_main.transform(X[:, 0])
        X[:, 1] = le_employ.transform(X[:, 1])
        X[:, 2] = le_country.transform(X[:, 2])
        X[:, 3] = le_education.transform(X[:, 3])
        X[:, 5] = le_sex.transform(X[:, 5])

        X = X.astype(float)

        salary = reg.predict(X)

        st.subheader(f'The Estimated Salary is ${salary[0]:.2f}')
