import time

import torch
import wandb

from debug_configuration import DEBUG_LOGGING, FLUSH_VAD


class VADIterator:
    def __init__(
            self,
            model,
            threshold: float = 0.5,
            sampling_rate: int = 16000,
            min_silence_duration_ms: int = 100,
            speech_pad_ms: int = 30,
            flush_timeout_ms: int = 5000  # Flush timeout in milliseconds (5 seconds)
    ):
        """
        VADIterator for streaming audio chunks.

        Parameters:
            model: Preloaded .jit/.onnx Silero VAD model.
            threshold: Speech threshold. Probabilities ABOVE this value indicate speech.
            sampling_rate: Supported sample rates: 8000 or 16000.
            min_silence_duration_ms: Minimum silence duration (ms) to consider end of speech.
            speech_pad_ms: Padding (ms) to add to final speech chunks.
            flush_timeout_ms: If no speech is detected for this period (ms), force flush of the buffer.
        """
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.buffer = []

        if sampling_rate not in [8000, 16000]:
            raise ValueError("VADIterator does not support sampling rates other than [8000, 16000]")

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.flush_timeout_samples = sampling_rate * flush_timeout_ms / 1000

        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False  # Indicates if an utterance is currently being recorded.
        self.temp_end = 0
        self.current_sample = 0
        self.last_speech_sample = 0  # Sample index when speech was last detected.

    @torch.no_grad()
    def __call__(self, x):
        """
        Processes an audio chunk and returns the buffered utterance when either:
          - End-of-speech is detected via short silence.
          - No speech is detected for a duration exceeding flush_timeout.

        Parameters:
            x: torch.Tensor containing the audio chunk.

        Returns:
            A list of audio chunks (the utterance) or None if still accumulating.
        """
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be cast to tensor. Cast it manually.")

        # Update current sample count based on the chunk length.
        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        # Compute speech probability for the current chunk.
        start_time = time.time()
        speech_prob = self.model(x, self.sampling_rate).item()
        end_time = time.time()
        latency = end_time - start_time
        if DEBUG_LOGGING:
            wandb.log({"VAD_latency": latency})

        # Update last_speech_sample if speech is detected.
        if speech_prob >= self.threshold:
            self.last_speech_sample = self.current_sample

        # If speech is detected and we haven't started an utterance, trigger a new one.
        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            self.buffer = []  # Start fresh.
            return None

        if self.triggered:
            # Accumulate the audio chunk.
            self.buffer.append(x)

            # Check flush timeout: if no speech is detected for longer than flush_timeout, flush the buffer.
            if (self.current_sample - self.last_speech_sample) > self.flush_timeout_samples:
                # Force flush due to prolonged silence.
                self.triggered = False
                utterance = self.buffer
                self.buffer = []
                self.temp_end = 0
                return utterance

            # Check normal end-of-speech detection using a minimum silence duration.
            if speech_prob < self.threshold - 0.15:
                if not self.temp_end:
                    self.temp_end = self.current_sample
                if (self.current_sample - self.temp_end) >= self.min_silence_samples:
                    self.triggered = False
                    utterance = self.buffer
                    self.buffer = []
                    self.temp_end = 0
                    return utterance
            else:
                # If speech resumes, reset the temporary silence counter.
                self.temp_end = 0

            # Additional safeguard: if the buffer grows too large, flush it.
            if len(self.buffer) >= 32 * FLUSH_VAD:
                self.triggered = False
                utterance = self.buffer
                self.buffer = []
                self.temp_end = 0
                return utterance

        return None
