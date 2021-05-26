import streamlit as st
import utils


def render():
    st.title("Let's clone a voice!")

    with st.sidebar:
        voice_file = st.file_uploader(
            "Voice to clone",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="A recording of one person speaking. No noises or music please!",
        )
        transcriber = utils.select_transcriber(st.empty())
        split_sentences = st.checkbox(
            "Pause between sentences",
            value=False,
            help="Might sound more natural, but will probably gobble up words.",
        )
    if voice_file is None:
        st.info("Uplaod a voice to get started! Use the sidebar to the left.")
        return

    voice_embedding = utils.embed_voice(voice_file)
    text = st.text_area(
        "Text to read",
        value="Have fun - but use protection! It's probably illegal to impersonate people.",
        help="Long text will not take much more time to read, so this can be as long as you like.",
    )

    if not st.button("Read it!"):
        return

    spectrogram = utils.text_to_spectrogram(
        transcriber(text), voice_embedding, split_sentences=split_sentences
    )
    audio = utils.spectrogram_to_audio(spectrogram, progress_bar=st.progress(0.0))

    left, right = st.beta_columns([3, 1])
    utils.play_audio(audio, widget=left.empty())
    utils.download_audio(audio, label="Click to download", widget=right.empty())


render()
