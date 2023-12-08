import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Data visualization')

st.header('Upload data file')
data_file = st.file_uploader('Choose a csv file', type=(['.csv']))

if data_file is not None:
  df = pd.read_csv(data_file)

  st.header('Show data')
  st.dataframe(df)

  st.header('Descriptive statistics')
  st.table(df.describe())

  st.header('Show data information')
  buffer = io.StringIO()
  df.info(buf=buffer)
  st.text(buffer.getvalue())

  st.header('Visualize each attribute')
  for col in list(df.columns):
    fig, ax = plt.subplots()
    ax.hist(df[col], bins=20)
    plt.xlabel(col)
    plt.ylabel('Quantity')
    st.pyplot(fig)

  st.header('Show correlation between variables')
  fig, ax = plt.subplots()
  sns.heatmap(df.corr(method='pearson'), ax=ax, vmax=1, square=True, annot=True, cmap='Reds')
  st.write(fig)

  output = st.radio('Choose a dependent variable', df.columns)

  st.header('Show relationship between variables')
  for col in list(df.columns):
    if col != output:
      fig, ax = plt.subplots()
      ax.scatter(x=df[col], y=df[output])
      plt.xlabel(col)
      plt.ylabel(output)
      st.pyplot(fig)
