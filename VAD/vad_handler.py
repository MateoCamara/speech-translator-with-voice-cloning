import logging
import numpy as np
import torch

from VAD.vad_iterator import VADIterator
from baseHandler import BaseHandler
from utils.utils import int2float

# Configure logger
logger = logging.getLogger(__name__)


class VADHandler(BaseHandler):
    """Handles voice activity detection by accumulating audio during speech."""

    def setup(
        self,
        should_listen,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=500,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
    ):
        # Initialize VAD parameters and model
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms

        # Load pre-trained VAD model from torch hub
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def process(self, audio_chunk):
        # Convert audio bytes to int16 numpy array
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        # Convert int16 array to float32 array
        audio_float32 = int2float(audio_int16)
        # Run voice activity detection
        vad_output = self.iterator(torch.from_numpy(audio_float32))

        if vad_output:
            logger.debug("VAD: end of speech detected")
            # Concatenate detected audio segments and convert to numpy array
            audio_array = torch.cat(vad_output).cpu().numpy()
            duration_sec = len(audio_array) / self.sample_rate
            duration_ms = duration_sec * 1000

            # Check if the audio duration is within the acceptable range
            if not (self.min_speech_ms <= duration_ms <= self.max_speech_ms):
                logger.debug(
                    f"Audio duration {duration_sec:.2f}s not in valid range, skipping"
                )
            else:
                yield audio_array

    @property
    def min_time_to_debug(self):
        # Return minimal debug time threshold
        return 1e-5
