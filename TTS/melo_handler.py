import os
import time
import logging

# import wandb
import torch
import numpy as np
import librosa
from melo.api import TTS
from baseHandler import BaseHandler
from rich.console import Console
from debug_configuration import DEBUG_LOGGING

logger = logging.getLogger(__name__)
console = Console()

# Mapping from Whisper language codes to Melo formats
LANG_MAP = {
    "en": "EN",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}

SPEAKER_MAP = {
    "en": "EN-BR",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}


class MeloTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="mps",
        language="en",
        speaker_to_id="en",
        gen_kwargs={},  # Unused
        blocksize=512,
        ckpt_path=None,
        config_path=None,
    ):
        self.should_listen = should_listen
        self.device = device
        self.language = language
        self.ckpt_path = ckpt_path
        self.config_path = config_path # or os.path.join(os.path.dirname(ckpt_path), "config.json")
        self.model = TTS(
            language=LANG_MAP[self.language],
            device=device,
            ckpt_path=self.ckpt_path,
            config_path=self.config_path,
        )
        self.speaker_id = self.model.hps.data.spk2id[SPEAKER_MAP[speaker_to_id]]
        self.blocksize = blocksize
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        # Generate a dummy output for warmup
        _ = self.model.tts_to_file("text", self.speaker_id, quiet=True)

    def process(self, llm_sentence):
        # Return silent block if no input
        if llm_sentence is None:
            return np.zeros(self.blocksize, dtype=np.int16)

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        if self.device == "mps":
            start = time.time()
            torch.mps.synchronize()  # Wait for MPS kernels to finish
            torch.mps.empty_cache()  # Free MPS memory
            _ = time.time() - start  # Required for stability

        try:
            start_time = time.time()
            audio_chunk = self.model.tts_to_file(llm_sentence, self.speaker_id, quiet=True)
            latency = time.time() - start_time
            # if DEBUG_LOGGING:
            #     wandb.log({"Melo_latency": latency})
        except (AssertionError, RuntimeError) as e:
            logger.error(f"Error in MeloTTSHandler: {e}")
            audio_chunk = np.array([])

        if not audio_chunk.size:
            self.should_listen.set()
            return

        # Resample audio from 44100 Hz to 16000 Hz and convert to int16
        audio_chunk = librosa.resample(audio_chunk, orig_sr=44100, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)

        # Yield fixed-size audio blocks with padding if necessary
        for i in range(0, len(audio_chunk), self.blocksize):
            block = audio_chunk[i : i + self.blocksize]
            if len(block) < self.blocksize:
                block = np.pad(block, (0, self.blocksize - len(block)))
            yield block

        # Yield three silent blocks at the end
        for _ in range(3):
            yield np.zeros(self.blocksize, dtype=np.int16)

        self.should_listen.set()
