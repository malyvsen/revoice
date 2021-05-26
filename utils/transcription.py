import streamlit as st
from transcriber import Transcriber


def select_transcriber(widget):
    choices = {
        "English (best)": None,
        "Polish": "pol-Latn",
        "Dutch": "nld-Latn",
        "Italian": "ita-Latn",
        "German (worst)": "deu-Latn-nar",
    }
    choice = widget.selectbox("Language", list(choices.keys()))
    return get_transcriber_by_code(choices[choice])


@st.cache()
def get_transcriber_by_code(code):
    if code is None:
        return lambda text: text
    return Transcriber.from_language_code(code).transcribe
