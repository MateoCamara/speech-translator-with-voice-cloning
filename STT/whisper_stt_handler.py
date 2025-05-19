import time
import logging
import torch
# import wandb
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from baseHandler import BaseHandler
from rich.console import Console
from debug_configuration import DEBUG_LOGGING

logger = logging.getLogger(__name__)
console = Console()

SUPPORTED_LANGUAGES = ["en", "fr", "es", "zh", "ja", "ko"]


class WhisperSTTHandler(BaseHandler):
    """Handles Speech-To-Text using a Whisper model."""

    def setup(
        self,
        model_name="distil-whisper/distil-large-v3",  # or "openai/whisper-large-v3-turbo"
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        language=None,
        gen_kwargs={},
    ):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.compile_mode = compile_mode
        self.gen_kwargs = gen_kwargs.copy()
        self.start_language = language
        self.last_language = language if language != "auto" else None

        if self.last_language:
            self.gen_kwargs["language"] = self.last_language

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=self.torch_dtype
        ).to(device)

        # Compile model if mode is provided
        if self.compile_mode:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = torch.compile(
                self.model.forward, mode=self.compile_mode, fullgraph=True
            )
        self.warmup()

    def prepare_model_inputs(self, spoken_prompt):
        # Convert spoken prompt to model input features
        input_features = self.processor(
            spoken_prompt, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device, dtype=self.torch_dtype)
        return input_features

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        n_steps = 1 if self.compile_mode == "default" else 2

        dummy_input = torch.randn(
            (1, self.model.config.num_mel_bins, 3000),
            dtype=self.torch_dtype,
            device=self.device,
        )

        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()

        for _ in range(n_steps):
            _ = self.model.generate(dummy_input, **self.gen_kwargs)

        if self.device == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) * 1e-3
            logger.info(f"{self.__class__.__name__} warmed up in {elapsed:.3f}s")

    def process(self, spoken_prompt):
        print("Inferring Whisper...")
        start_time = time.time()

        input_features = self.prepare_model_inputs(spoken_prompt)
        pred_ids = self.model.generate(input_features, **self.gen_kwargs)

        pred_text = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True, decode_with_timestamps=False
        )[0]
        language_code = self.processor.tokenizer.decode(pred_ids[0, 1])[2:-2]  # remove "<|" and "|>"

        logger.debug("Finished Whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Whisper: {language_code}")

        if self.start_language == "auto":
            language_code += "-auto"

        latency = time.time() - start_time

        if DEBUG_LOGGING:
            # wandb.log({"Whisper_latency": latency})
            with open("./tests/latency/whisper_pred.txt", "a", encoding="utf-8") as f:
                f.write(" " + pred_text)

        yield (pred_text, language_code)
