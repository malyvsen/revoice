import streamlit as st
import requests
from pathlib import Path
import zipfile
import tempfile
from io import BytesIO
import numpy as np
import librosa
from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

from .audio import AudioLengthError
from .text import supersplit


encoder_path = Path("encoder/saved_models/pretrained.pt")
synthesizer_path = Path("synthesizer/saved_models/pretrained/pretrained.pt")
vocoder_path = Path("vocoder/saved_models/pretrained/pretrained.pt")


def prepare_models():
    if not all(
        path.exists() for path in [encoder_path, synthesizer_path, vocoder_path]
    ):
        response = requests.get(
            "https://github.com/blue-fish/Real-Time-Voice-Cloning/releases/download/v1.0/pretrained.zip"
        )
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        zip_file.extractall("./")
    encoder.load_model(encoder_path)
    vocoder.load_model(vocoder_path)


def get_synthesizer():
    return Synthesizer(synthesizer_path)


def text_to_spectrogram(text, voice_embedding, split_sentences: bool):
    sentence_endings = ".?!\n" if split_sentences else "\n"
    sentences = [text]
    for ending in sentence_endings:
        sentences = [
            sentence.strip()
            for bundle in sentences
            for sentence in supersplit(bundle, ending)
        ]

    spectrograms = get_synthesizer().synthesize_spectrograms(
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
