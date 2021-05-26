import streamlit as st
import utils


def render():
    st.title("Let's clone a voice!")
    st.text("Have fun - but stay safe! Impersonating people is (probably) illegal.")
    voice_file = st.file_uploader(
        "Voice to clone",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
        help="A recording of one person speaking. No noises or music please!",
    )
    if voice_file is None:
        return
    voice_embedding = utils.embed_voice(voice_file)
    text = st.text_area(
        "Text to read",
        help="Long text will not take much more time to read, so this can be as long as you like.",
    )

    transcriber = utils.select_transcriber(st.empty())
    if not st.button("Read it!"):
        return

    spectrogram = utils.text_to_spectrogram(transcriber(text), voice_embedding)
    audio = utils.spectrogram_to_audio(spectrogram, progress_bar=st.progress(0.0))

    left, right = st.beta_columns([3, 1])
    utils.play_audio(audio, widget=left.empty())
    utils.download_audio(audio, label="Click to download", widget=right.empty())


render()
