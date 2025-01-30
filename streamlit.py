import tensorflow as tf
import streamlit as st
import warnings
import PIL
import os
import io
import numpy as np
from tensorflow.keras import layers
warnings.filterwarnings('ignore')

# Load the model
model=tf.keras.models.load_model("basemodel.h5")    

st.set_page_config(layout='wide')

st.markdown("<h1 style ='text-align: center;color: red'>**Ninja Cart Classification Project**</h1>",unsafe_allow_html=True)

img=st.file_uploader("Please upload the image of Ninja Cart Vegetables in JPG or PNG format",type=['jpg','png'])

class_label=['indian market', 'onion', 'potato', 'tomato']

st.text(f"List of class labels the model can predict: {', '.join(class_label)}")

if img is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        
        # Image to be Predicted
        
        """
        )
    image=PIL.Image.open(img)
    st.image(image)
    arr=np.array(image)
    data_processing=tf.keras.Sequential([layers.Rescaling(1.0/255),layers.Resizing(height=224,width=224)])
    input=data_processing(arr)

    y_probs=model.predict(tf.expand_dims(input, axis=0))

    with col2:

        if(st.button("Predict")):

            prediction=class_label[np.argmax(y_probs)]

            st.write(f"Model Has predicted the image as {prediction}")
    
else:
    st.text("Please upload the image")

st.markdown("<h1 style ='text-align: center;color: red'>**Below is the Model Architecture**</h1>",unsafe_allow_html=True)

# To Print the Summary info in STreamlit
summary_str = io.StringIO()
model.summary(print_fn=lambda x: summary_str.write(x + "\n"))

# Display in Streamlit
st.text("### Model Summary")
st.code(summary_str.getvalue(), language="plaintext")

