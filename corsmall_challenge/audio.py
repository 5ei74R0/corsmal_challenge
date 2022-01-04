import numpy


class AudioProcessor:
    def __init__(
        self, sample_rate, signal, frame_length_t=0.025, frame_stride_t=0.01, nfilt=64
    ):

        self.sample_rate = sample_rate
        self.signal = signal
        self.frame_length_t = frame_length_t
        self.frame_stride_t = frame_stride_t
        self.signal_length_t = float(signal.shape[0] / sample_rate)
        self.frame_length = int(
            round(frame_length_t * sample_rate)
        )  # number of samples
        self.frame_step = int(round(frame_stride_t * sample_rate))
        self.signal_length = signal.shape[0]
        self.nfilt = nfilt
        self.num_frames = int(
            numpy.ceil(
                float(numpy.abs(self.signal_length - self.frame_length)) / self.frame_step
            )
        )
        self.pad_signal_length = self.num_frames * self.frame_step + self.frame_length
        self.NFFT = 512
