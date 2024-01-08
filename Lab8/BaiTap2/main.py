import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

class_list = {'0': 'Normal', '1': 'Pneumonia'}

st.title('Pneumonia Detection')

import os

script_directory = os.path.dirname(os.path.realpath(__file__))

file_path = os.path.join(script_directory, 'lrc_xray.pkl')

if os.path.exists(file_path):
    with open(file_path, 'rb') as input:
        
        model = pkl.load(input)
        
        st.header('Upload an image')
        image = st.file_uploader('Choose an image', type=(['png', 'jpg', 'jpeg']))
        
        if image is not None:
          image = Image.open(image)
          st.image(image, caption='Test image')
        
          if st.button('Predict'):
            image = image.resize((227*227*3, 1))
            vector = np.array(image)
            label = str((model.predict(vector))[0])
        
            st.header('Result')
            st.text(class_list[label])
else:
    print(f"File not found: {file_path}")

