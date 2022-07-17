from importlib.resources import path
import streamlit as st
from text.contents import *

# images load
st.markdown('## images load')
st.code(breed_list, language='python')

# how many breeds and pictures we have
st.markdown('## how many breeds')
st.code(breeds, language='python')

st.markdown('## how many pictures')
st.code(images, language='python')

# label strings and numbers mapping
st.markdown('## label strings and numbers mapping')
st.code(labeling, language='python')

# show some pic
st.markdown('### show some pic')
st.code(show_pic, language='python')
st.code(print_pic, language='python')

# Crop and save pictures
st.markdown('### Crop and save pictures')
st.code(crop_save, language='python')

# prepare data for training
st.markdown('### prepare data for training')
st.code(paths_and_labels, language='python')

# set class
st.markdown('### set class')
st.code(set_class, language='python')

# split train, data
st.markdown('### split train, data')
st.code(split_data, language='python')

# set_layer
st.markdown('### set_layer')
st.code(set_layer, language='python')

# compile
st.markdown('### compile')
st.code(compile_model, language='python')

# fitting
st.markdown("### fit_generator")
st.code(fitting, language='python')
