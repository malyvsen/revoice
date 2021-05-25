import streamlit as st
import base64
from pathlib import Path
import tempfile
import numpy as np
import librosa
import soundfile
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder
from transcriber import Transcriber


def init():
    global synthesizer
    encoder.load_model(Path("encoder/saved_models/pretrained.pt"))
    synthesizer = Synthesizer(Path("synthesizer/saved_models/pretrained/pretrained.pt"))
    vocoder.load_model(Path("vocoder/saved_models/pretrained/pretrained.pt"))


def synthesize_audio(text, voice_embedding, progress_bar, rerun=False):
    @st.cache(suppress_st_warning=True, max_entries=1)
    def slave(text, voice_embedding, progress_bar, run_id):
        spectrogram = text_to_spectrogram(text, voice_embedding)
        return spectrogram_to_audio(spectrogram, progress_bar=progress_bar)

    return slave(
        text, voice_embedding, progress_bar, None if rerun else np.random.rand()
    )


def text_to_spectrogram(text, voice_embedding):
    sentence_endings = ".?!\n"
    sentences = [text]
    for ending in sentence_endings:
        sentences = [
            sentence.strip()
            for bundle in sentences
            for sentence in supersplit(bundle, ending)
        ]

    spectrograms = synthesizer.synthesize_spectrograms(
        sentences, [voice_embedding] * len(sentences)
    )
    return np.concatenate(spectrograms, axis=1)


def spectrogram_to_audio(spectrogram, progress_bar):
    def progress_callback(num_completed, total, batch_size, generation_rate):
        progress_bar.progress(num_completed / total)

    result = vocoder.infer_waveform(spectrogram, progress_callback=progress_callback)
    progress_bar.empty()
    return result


@st.cache(max_entries=64)
def embed_voice(file):
    with tempfile.NamedTemporaryFile("w+b") as temp_file:
        temp_file.write(file.read())
        audio, sample_rate = librosa.load(temp_file.name, mono=True)
    AudioLengthError.raise_maybe(audio, sample_rate)
    preprocessed = encoder.preprocess_wav(audio, sample_rate)
    return encoder.embed_utterance(preprocessed)


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


def play_audio(audio, widget, sample_rate=None):
    widget.audio(audio_bytes(audio, sample_rate), format="audio/wav")


def download_audio(audio, label: str, widget, filename="speech.wav", sample_rate=None):
    b64 = base64.b64encode(audio_bytes(audio, sample_rate=sample_rate)).decode()
    widget.markdown(
        f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{label}</a>',
        unsafe_allow_html=True,
    )


def audio_bytes(audio, sample_rate=None):
    if sample_rate is None:
        sample_rate = encoder.sampling_rate
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / f"audio.wav"
        soundfile.write(file_path, audio, samplerate=sample_rate)
        with open(file_path, "rb") as file:
            return file.read()


class AudioLengthError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    @classmethod
    def raise_maybe(cls, audio, sample_rate, min_length=4, max_length=60):
        if len(audio) / sample_rate < min_length:
            raise cls(f"Audio is too short - try something over {min_length} seconds")
        elif len(audio) / sample_rate > max_length:
            raise cls(f"Audio is too long - try something under {max_length} seconds")


def supersplit(string: str, delimiter: str):
    """Like str.split, but keeps delimiter and discards empty bits."""
    return [
        bit
        for split in string.split(delimiter)
        for bit in [delimiter, split]
        if len(bit) > 0
    ][1:]
