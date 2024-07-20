import streamlit as st
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import PyPDF2
import plotly.graph_objects as go
from Courses import ds_course
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import time

# Load pre-trained model
model = load('model.joblib')

# Define title style
title_style = """
    color: #2e6bb3; 
    font-size: 48px; 
    font-weight: bold; 
    font-family: Arial, sans-serif; 
    text-align: center; 
    padding-bottom: 20px;
"""

# Display title
st.markdown(
    f'<h1 style="{title_style}">Hello dear candidate In Your App Skills matcher</h1>',
    unsafe_allow_html=True
)
st.markdown('---')

# Sidebar elements
st.sidebar.image("data/images/CA.png", use_column_width=True)
st.sidebar.markdown('<p style="font-size: 16px; color: #555; text-align: center;">Created by Said Tallouk</p>', unsafe_allow_html=True)

# Input fields for job offer text and CV upload
offer_text = st.text_area("Step 1: Paste the job offer text here", height=200)
cv_file = st.file_uploader("Step 2: Upload your CV (PDF file)", key="cv_file")
st.markdown('---')

def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text += page.extract_text()
    return text

def calculate_similarity(offer_text, cv_text):
    offer_tokens = word_tokenize(offer_text.lower())
    cv_tokens = word_tokenize(cv_text.lower())

    offer_vectors = np.array([model.wv.get_vector(token) for token in offer_tokens if token in model.wv.key_to_index])
    cv_vectors = np.array([model.wv.get_vector(token) for token in cv_tokens if token in model.wv.key_to_index])

    if len(offer_vectors) == 0 or len(cv_vectors) == 0:
        return 0  # If no valid vectors found, return 0 similarity

    offer_vector = np.mean(offer_vectors, axis=0)
    cv_vector = np.mean(cv_vectors, axis=0)

    similarity_score = cosine_similarity([offer_vector], [cv_vector])[0][0]
    similarity_score = max(similarity_score, 0)
    return similarity_score

cv_text = ""  # Placeholder for CV text, to be updated once the user uploads their CV
similarity_score = None  # Initialize similarity_score
st.markdown('Step 3: When ready click the Scan button.')

col1, col2 = st.columns([1, 100])
centered_button_container = col2.empty()
with centered_button_container:
    st.write("")  # Placeholder to center-align the button

if centered_button_container.button('Scan') and offer_text.strip() != "" and cv_file is not None:
    cv_text = extract_text_from_pdf(cv_file)
    similarity_score = calculate_similarity(offer_text, cv_text)
    st.write(f'Similarity with job offer: {similarity_score:.2f}')
    
    # Generate word clouds
    # if offer_text:
        
    #     st.markdown('''<hr style="border:2px solid #cccccc; margin: 20px 0;" />''', unsafe_allow_html=True)
    #     st.markdown('<h3 style="text-align: center; color: #2e6bb3;">Word Cloud for Job Offer', unsafe_allow_html=True)
    #     st.markdown('''<hr style="border:2px solid #cccccc; margin: 20px 0;" />''', unsafe_allow_html=True)
    #     wordcloud_offer = WordCloud(width=800, height=400, background_color='white').generate(offer_text)
    #     fig_offer, ax_offer = plt.subplots()
    #     ax_offer.imshow(wordcloud_offer, interpolation='bilinear')
    #     ax_offer.axis('off')
    #     st.pyplot(fig_offer)

    # if cv_text:
    #     st.markdown('''<hr style="border:2px solid #cccccc; margin: 20px 0;" />''', unsafe_allow_html=True)
    #     st.markdown('<h3 style="text-align: center; color: #2e6bb3;">Word cloud pour CV</h3>', unsafe_allow_html=True)
    #     st.markdown('''<hr style="border:2px solid #cccccc; margin: 20px 0;" />''', unsafe_allow_html=True)
    #     wordcloud_cv = WordCloud(width=800, height=400, background_color='white').generate(cv_text)
    #     fig_cv, ax_cv = plt.subplots()
    #     ax_cv.imshow(wordcloud_cv, interpolation='bilinear')
    #     ax_cv.axis('off')
    #     st.pyplot(fig_cv)
    # Ajouter des lignes de séparation et un titre stylisé
    # st.markdown('''<hr style="border:2px solid #cccccc; margin: 20px 0;" />''', unsafe_allow_html=True)
    # st.markdown('<h3 style="text-align: center; color: #2e6bb3;">Correspondance entre l\'offre et le CV</h3>', unsafe_allow_html=True)
    # st.markdown('''<hr style="border:2px solid #cccccc; margin: 20px 0;" />''', unsafe_allow_html=True)

    # Gauge chart with animation
    if similarity_score < 0.5:
        bar_color = 'red'
        emoji = "😔" 
        message = "Votre candidature a été rejetée pour cette offre d'emploi."
    else:
        bar_color = 'green'
        emoji = "😊"
        message = "Félicitations! Votre candidature a été acceptée pour cette offre d'emploi."

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match with offer"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': bar_color}, 
            'steps': [
                {'range': [0, 10], 'color': "lightgray"},
                {'range': [10, 30], 'color': "lightgray"},
                {'range': [30, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "lightgray"},
                {'range': [70, 90], 'color': "lightgray"},
                {'range': [90, 100], 'color': "lightgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': similarity_score * 100}},
        number={'suffix': "%" 
        }
    ))

    chart = st.empty()
    progress = 0
    while progress <= similarity_score * 100:
        time.sleep(0.1)
        progress += 1
        fig.update_traces(value=progress)
        chart.plotly_chart(fig, use_container_width=True)
    
    st.write(f"{emoji} {message}")

    # Pie chart
    pie_chart = st.empty()
    progress_pie = 0
    while progress_pie <= similarity_score:
        time.sleep(0.1)
        progress_pie += 0.01
        fig_pie_similarity = go.Figure(go.Pie(
            labels=["Similar", "Not Similar"],
            values=[progress_pie, 1 - progress_pie],
            hole=0.3,
            marker=dict(colors=[bar_color, 'lightgray'])
        ))
        fig_pie_similarity.update_layout(title='Similarity with offer')
        pie_chart.plotly_chart(fig_pie_similarity)

    # Line chart for similarity score
    x_values = list(range(0, 101))
    y_values = [similarity_score * (i / 100) for i in x_values]

    fig_line = go.Figure(data=go.Scatter(x=x_values, y=y_values, mode='lines+markers', line=dict(color=bar_color)))
    fig_line.update_layout(
        title="Évolution de la similarité avec l'offre",
        xaxis_title="Étapes de l'analyse",
        yaxis_title="Score de similarité",
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 1])
    )

    st.plotly_chart(fig_line)

if similarity_score is not None and similarity_score < 0.5:
    st.success("**Pour renforcer votre CV en vue de notre prochaine offre d'emploi, nous vous recommandons de vous concentrer sur les domaines suivants, étroitement liés à nos futures offres d'emploi :**")
    
    # Create an ordered list of recommended courses
    recommended_courses_list = [f"{i+1}. [{course[0]}]({course[1]})" for i, course in enumerate(ds_course)]
    
    # Split the list into groups of four
    grouped_courses = [recommended_courses_list[i:i+4] for i in range(0, len(recommended_courses_list), 4)]
    
    # Display each group of recommended courses
    for group in grouped_courses:
        recommended_courses_str = "\n".join(group)
        st.markdown(recommended_courses_str)
        st.markdown('---')
