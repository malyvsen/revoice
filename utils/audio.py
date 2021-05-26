import base64
from pathlib import Path
import tempfile
import soundfile
from encoder import inference as encoder


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
