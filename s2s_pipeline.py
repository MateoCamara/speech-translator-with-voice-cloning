import json
import logging
import os
from pathlib import Path
from queue import Queue
from threading import Event

import torch
import nltk
from rich.console import Console
import wandb

from debug_configuration import USE_COMPLETION_LLM

# Initialize wandb project
wandb.init(project="latency-tests", mode="online")

# Ensure necessary NLTK resources are available
try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, OSError):
    nltk.download("punkt_tab")
try:
    nltk.data.find("tokenizers/averaged_perceptron_tagger_eng")
except (LookupError, OSError):
    nltk.download("averaged_perceptron_tagger_eng")

# Set caching directory for torch compiler optimizations
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")

console = Console()
logging.getLogger("numba").setLevel(logging.WARNING)  # Reduce numba logs

# Import only the necessary argument classes
from utils.thread_manager import ThreadManager

# ---------------- Configuration Loading ----------------
def load_config():
    """
    Load configuration from 'config.json' in the project root.
    The file should be edited by the user with appropriate values.
    Dummy default values are provided.
    """
    config_path = Path(__file__).resolve().parent / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)

def wrap_config(data):
    """
    Convert a dictionary to an object with attribute-style access.
    """
    return type("Config", (), data)


def process_gen_params(params):
    base_params = {}
    gen_kwargs = {}
    for key, value in params.items():
        if key.startswith("gen_"):
            # Remover el prefijo "gen_" y agregar al diccionario de gen_kwargs
            new_key = key[4:]
            gen_kwargs[new_key] = value
        else:
            base_params[key] = value
    # Agregar el diccionario de gen_kwargs a los parámetros base
    base_params["gen_kwargs"] = gen_kwargs
    return base_params

# ---------------- Logger Setup ----------------
def setup_logger(log_level):
    """
    Configure the root logger.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    global logger
    logger = logging.getLogger(__name__)
    if log_level.lower() == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

# ---------------- Queues and Events ----------------
def initialize_queues_and_events():
    """
    Initialize all queues and threading events for inter-thread communication.
    """
    return {
        "stop_event": Event(),
        "should_listen": Event(),
        "recv_audio_chunks_queue": Queue(),
        "send_audio_chunks_queue": Queue(),
        "spoken_prompt_queue": Queue(),
        "text_prompt_queue": Queue(),
        "lm_response_queue": Queue(),
    }

# ---------------- Pipeline Building ----------------
def build_pipeline(module_kwargs,
                   vad_handler_kwargs,
                   whisper_stt_handler_kwargs,
                   language_model_handler_kwargs,
                   melo_tts_handler_kwargs,
                   queues_and_events):
    """
    Build the production pipeline with local audio streaming, VAD, Whisper STT,
    Transformers language model and Melo TTS.
    """
    stop_event = queues_and_events["stop_event"]
    should_listen = queues_and_events["should_listen"]
    recv_audio_chunks_queue = queues_and_events["recv_audio_chunks_queue"]
    send_audio_chunks_queue = queues_and_events["send_audio_chunks_queue"]
    spoken_prompt_queue = queues_and_events["spoken_prompt_queue"]
    text_prompt_queue = queues_and_events["text_prompt_queue"]
    lm_response_queue = queues_and_events["lm_response_queue"]

    # Import required handlers (assumed to exist in their respective modules)
    from connections.local_audio_streamer import LocalAudioStreamer
    from VAD.vad_handler import VADHandler
    from STT.whisper_stt_handler import WhisperSTTHandler
    if USE_COMPLETION_LLM:
        from LLM.language_model_with_completion import LanguageModelHandler
    else:
        from LLM.language_model import LanguageModelHandler
    from TTS.melo_handler import MeloTTSHandler

    # Local audio streamer for input/output
    local_audio_streamer = LocalAudioStreamer(
        input_queue=recv_audio_chunks_queue,
        output_queue=send_audio_chunks_queue
    )
    # Voice Activity Detection handler
    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vad_handler_kwargs
    )
    # Whisper Speech-to-Text handler
    stt = WhisperSTTHandler(
        stop_event,
        queue_in=spoken_prompt_queue,
        queue_out=text_prompt_queue,
        setup_kwargs=process_gen_params(whisper_stt_handler_kwargs)
    )
    # Transformers Language Model handler
    lm = LanguageModelHandler(
        stop_event,
        queue_in=text_prompt_queue,
        queue_out=lm_response_queue,
        setup_kwargs=process_gen_params(language_model_handler_kwargs)
    )
    # Melo Text-to-Speech handler
    tts = MeloTTSHandler(
        stop_event,
        queue_in=lm_response_queue,
        queue_out=send_audio_chunks_queue,
        setup_args=(should_listen,),
        setup_kwargs=melo_tts_handler_kwargs
    )

    print("READY")
    # Return a ThreadManager that will handle all threads in the pipeline
    return ThreadManager([local_audio_streamer, vad, stt, lm, tts])

# ---------------- Main Function ----------------
def main():
    # Load configuration from file
    config = load_config()

    # Wrap each configuration section to access attributes
    module_kwargs = config.get("module", {})
    vad_handler_kwargs = config.get("vad_handler", {})
    whisper_stt_handler_kwargs = config.get("whisper_stt_handler", {})
    language_model_handler_kwargs = config.get("language_model_handler", {})
    melo_tts_handler_kwargs = config.get("melo_tts_handler", {})

    setup_logger(module_kwargs["log_level"])
    queues_and_events = initialize_queues_and_events()
    pipeline_manager = build_pipeline(
        module_kwargs,
        vad_handler_kwargs,
        whisper_stt_handler_kwargs,
        language_model_handler_kwargs,
        melo_tts_handler_kwargs,
        queues_and_events
    )

    try:
        pipeline_manager.start()
    except KeyboardInterrupt:
        pipeline_manager.stop()

# ---------------- Unit Tests ----------------
# Run tests by passing "test" as a command line argument.
if __name__ == "__main__":
    main()
