import streamlit as st
from fastai.vision.all import *
import pathlib
import os
import torch
import plotly.express as px

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
# torch.classes.__path__ = []


plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Transportni klassifikatsiya qiluvchi model!")

# rasmni joylash
f = st.file_uploader("Rasm yuklash", type=['png','jpeg','gif','svg'])
if f:
    st.image(f)

    # PIL convert
    img = PILImage.create(f)

    model = load_learner("transport_model.pkl")

    pred, pred_id, probs = model.predict(img)
    st.success(f"Prediction: {pred}")
    st.info(f"Accuracy: {probs[pred_id]*100:.1f}%")

    # plotting
    fig = px.bar(x=model.dls.vocab, y=probs*100)
    st.plotly_chart(fig)
