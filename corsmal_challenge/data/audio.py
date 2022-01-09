import pathlib
from typing import Tuple

import torch
import torchaudio


class Wav:
    def __init__(self, path: pathlib.Path):
        tup: Tuple[torch.Tensor, int] = torchaudio.load(path)
        wav_audio_data = tup[0]
        sample_rate = tup[1]
        self.wav_audio_data = wav_audio_data
        self.sample_rate = sample_rate

    def get_audio_length(self) -> float:
        """Return length of audio data

        Returns:
            float: length of audio data (sec)
        """
        return self.wav_audio_data.size()[1] / self.sample_rate

    def generate_mel_spectrogram(self, hop_length: int = 512, n_fft: int = 2048) -> torch.Tensor:
        """Generate mel spectrogram

        Args:
            hop_length (int, optional): hop length on time axis. Defaults to 512.
            n_fft (int, optional): window size on frequency axis. Defaults to 2048.

        Returns:
            torch.Tensor: tensor of mel spectrogram (channel, freq_dim, time_dim)
        """
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
        )(self.wav_audio_data)


def load_wav(path: pathlib.Path) -> Wav:
    """Given path to the WAV file, return tensor and sampling rate.

    Args:
        path (pathlib.Path): path to the WAV file

    Returns:
        Wav: sampled audio data & sampling rate
    """
    # return torchaudio.load(path)
    return Wav(path)
