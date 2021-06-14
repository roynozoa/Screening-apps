# Screening Apps Application
# Streamlit implementation

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
st.write("""
# COVID-19 Screening-apps

This app predicts the **COVID-19 Vaccination** Screening !!!



""")

# Loads dataset
data_df = pd.read_csv('data.csv')
data_df = data_df.iloc[:, 0:14]

# get vaccination info
time_data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/country_data/Indonesia.csv')
time_data.drop(columns=['source_url'], inplace=True)
time_data['date'] = pd.to_datetime(time_data['date'])
time_data.set_index('date', inplace=True)
time_data.drop(columns=['location', 'vaccine', 'total_vaccinations'], inplace=True)
latest_date = time_data.index[-1]

st.sidebar.header('User Input Features')

def user_input_features():
    terpapar = st.sidebar.selectbox('Pernah Terpapar COVID/positif dalam 3 bulan',('Yes','No'))
    u18 = st.sidebar.selectbox('Berusia kurang dari 18 tahun',('Yes','No'))
    mengandung = st.sidebar.selectbox('Ibu yang sedang mengandung',('Yes','No'))
    tekanandarah = st.sidebar.selectbox('Tekanan darah anda saat ini diatas 180/110 mmHg',('Yes','No'))
    menyusui = st.sidebar.selectbox('Ibu yang sedang menyusui',('Yes','No'))
    penyakit = st.sidebar.selectbox('Mengidap salah satu penyakit ini ( PPOK, Asma, Jantung, Gangguan Ginjal, penyakit hati)',('Yes','No'))
    alergi = st.sidebar.selectbox('Memiliki riwayat alergi terhadap vaksin',('Yes','No'))
    terapi = st.sidebar.selectbox('Sedang menjalani terapi kanker',('Yes','No'))
    autoimun = st.sidebar.selectbox('Mengidap penyakit autoimun sistemik',('Yes','No'))
    pembekuan = st.sidebar.selectbox('Mengidap gangguan pembekuan darah, defisiensi imun, atau penerima produk darah/transfusi',('Yes','No'))
    epilepsi = st.sidebar.selectbox('Mengidap penyakit epilepsi/ayan',('Yes','No'))
    vaksinlain = st.sidebar.selectbox('Mendapat vaksin lain(selain COVID) selama 1 bulan terakhir',('Yes','No'))
    hivaids = st.sidebar.selectbox('Mendiap HIV-AIDS',('Yes','No'))
    diatas60 = st.sidebar.selectbox('Berusia 60 tahun keatas',('Yes','No'))

    data = {'terpapar': terpapar,
            'u18': u18,
            'mengandung': mengandung,
            'tekanandarah': tekanandarah,
            'menyusui': menyusui,
            'penyakit': penyakit,
            'alergi': alergi,
            'terapi': terapi,
            'autoimun': autoimun,
            'pembekuan': pembekuan,
            'epilepsi': epilepsi,
            'vaksinlain': vaksinlain,
            'hivaids': hivaids,
            'diatas60': diatas60}

    features = pd.DataFrame(data, index=[0])
    
    data_cleanup = {'terpapar': {'Yes':1, 'No':0},
                    'u18': {'Yes':1, 'No':0},
                    'mengandung': {'Yes':1, 'No':0},
                    'tekanandarah': {'Yes':1, 'No':0},
                    'menyusui': {'Yes':1, 'No':0},
                    'penyakit': {'Yes':1, 'No':0},
                    'alergi': {'Yes':1, 'No':0},
                    'terapi': {'Yes':1, 'No':0},
                    'autoimun': {'Yes':1, 'No':0},
                    'pembekuan': {'Yes':1, 'No':0},
                    'epilepsi': {'Yes':1, 'No':0},
                    'vaksinlain': {'Yes':1, 'No':0},
                    'hivaids': {'Yes':1, 'No':0},
                    'diatas60': {'Yes':1, 'No':0}}

    features = features.replace(data_cleanup)
    return features

input_df = user_input_features()


# Reads in saved classification model
load_clf = pickle.load(open('clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)


st.header('Prediction')

vaccine = np.array(['Belum Bisa Divaksin', 'Bisa Divaksin'])
st.write(vaccine[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.write('---')
st.header('Vaccination Data in Indonesia')
st.write(f'Latest update {latest_date.strftime("%A, %d %B %Y")}')
st.write('''
Data source from [Our World in data](https://ourworldindata.org/coronavirus-source-data)

''')

st.line_chart(time_data)
# explainer = shap.TreeExplainer(load_clf)
# shap_values = explainer.shap_values(data_df)

# st.header('Feature Importance')
# plt.title('Feature Importance for COVID-19 Vaccine Screening')



# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, data_df, plot_type="bar")
# st.pyplot()

st.write('---')

st.write('''
For mor information about COVID-19 Vaccine in Indonesia please check Vaccine Dashboard from Kementerian Kesehatan RI
[link](https://vaksin.kemkes.go.id/#/vaccines) 

''')

st.write('---')



st.write("""
### Our Team :
- Muhammad Adisatriyo Pratama
- Stefannov
- Surya Asmoro

""")
