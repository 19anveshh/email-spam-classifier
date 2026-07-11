import streamlit as st
import pickle

# Load model and vectorizer once, cached across reruns
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("My Model Demo")

user_input = st.text_area("Enter text to classify:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]

        st.success(f"Prediction: {prediction}")

        # If your model supports probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            st.write("Confidence scores:")
            st.bar_chart(proba)