from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from IPython.display import YouTubeVideo


st.title('Text Summarization From Youtube Videos')
st.markdown('Using distilbart')

url = st.text_area('Paste the Youtube Video URL here')

def run_model(url):
    url_id = str.split(url,'/')[-1]
    st.video(url)
    srt = YouTubeTranscriptApi.get_transcript(url_id)
    txt = " "
    for elem in srt:
        txt += " " + elem['text']
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summary = summarizer(txt, max_length=130, min_length=30, do_sample=False)
    st.write('Summary')
    st.success(summary[-1]['summary_text'])

if st.button('Submit'):
    run_model(url)