import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import seaborn as sns
st.set_page_config(layout="wide")
# For EDA
df = pd.read_csv('data/Employee.csv')

st.title("Employee Future Prediction")
st.header("LIKELION AI SCHOOL-6th, 조용한사자처럼")
st.header("About the Data")
st.markdown('''
The data has following features : 
Independent value
- __Education__ : EDUCATION LEVEL
- __JoiningYear__ : YEAR OF JOINING COMPANY
- __City__ : CITY OFFICE WHERE POSTED
- __PaymentTier__ : PAYMENT TIER - 1: HIGHEST | 2: MID LEVEL | 3:LOWEST
- __Age__ : CURRENT AGE
- __Gender__ : GENDER OF EMPLOYEE
- __EverBenched__ : EVER KEPT OUT OF PROJECTS FOR 1 MONTH OR MORE
- __ExperienceInCurrentDomain__ : EXPERIENCE IN CURRENT FIELD
Target value
- __LeaveOrNot__ : WHETHER EMPLOYEE LEAVES THE COMPANY IN NEXT 2 YEARS
''')
st.markdown('##')
st.subheader('Dataset Sample')
st.write(df.head())

a = st.selectbox( 'Select Feature', ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain', 'LeaveOrNot'])
if a :
    desc = pd.DataFrame(df[a].describe()).T
    st.dataframe(desc)

st.markdown('##')
st.subheader('Check Graph')
b = st.sidebar.selectbox("확인하실 변수타입을 골라주세요",
                        ['연속형', '범주형'])
c = st.selectbox( 'Select Continuous Feature', ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain', 'LeaveOrNot'])
d = st.selectbox( 'Select Discrete Feature', ['Education', 'City', 'Gender', 'EverBenched'])
if b == '연속형':
    # fig = plt.figure(figsize = (20, 10))
    # sns.countplot(data = df, x = c )
    # st.pyplot(fig)
    aa = df.groupby(c).Education.count()
    fig = px.bar(data_frame = aa, x = aa.index, y = aa.values, color = aa.index)
    st.plotly_chart(fig)

elif b == '범주형' :
    # fig = plt.figure(figsize = (20, 10))
    # sns.histplot(data = df, x = d )
    # st.pyplot(fig)
    bb = df.groupby(d).Education.count()
    fig = px.bar(data_frame = bb, x = bb.index, y = bb.values, color = bb.index)
    st.plotly_chart(fig)
