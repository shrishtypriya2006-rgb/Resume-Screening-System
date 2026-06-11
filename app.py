import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("resume_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

categories = {
    0: 'Advocate',
    1: 'Arts',
    2: 'Automation Testing',
    3: 'Blockchain',
    4: 'Business Analyst',
    5: 'Civil Engineer',
    6: 'Data Science',
    7: 'Database',
    8: 'DevOps Engineer',
    9: 'DotNet Developer',
    10: 'ETL Developer',
    11: 'Electrical Engineering',
    12: 'HR',
    13: 'Hadoop',
    14: 'Health and fitness',
    15: 'Java Developer',
    16: 'Mechanical Engineer',
    17: 'Network Security Engineer',
    18: 'Operations Manager',
    19: 'PMO',
    20: 'Python Developer',
    21: 'SAP Developer',
    22: 'Sales',
    23: 'Testing',
    24: 'Web Designing'
}

st.title("Resume Screening System")

st.write("Enter a resume and predict its job category")

resume_text = st.text_area("Paste Resume Text Here")

if st.button("Predict Category"):
    if resume_text.strip():
        resume_vector = tfidf.transform([resume_text])
        prediction = model.predict(resume_vector)
        predicted_category = categories[int(prediction[0])]
        st.success(f"Predicted Category: {predicted_category}")
    else:
        st.warning("Please enter resume text")