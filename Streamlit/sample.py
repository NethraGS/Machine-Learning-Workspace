import streamlit as st

st.title("Hello Streamlit")

name = st.text_input("Enter your name:")
age = st.slider("Select your age:", 0, 100)

if st.button("Submit"):
    st.write(f"Hello {name}, you are {age} years old!")