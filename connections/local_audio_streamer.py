import threading
import sounddevice as sd
import numpy as np

import time
import logging

from debug_configuration import INPUT_DEVICES

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue,
        output_queue,
        list_play_chunk_size=512,
    ):
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue

    def run(self):
        def callback(indata, outdata, frames, time, status):
            self.input_queue.put(indata.copy())
            if not self.output_queue.empty():
                outdata[:] = self.output_queue.get()[:, np.newaxis]
                # print("Hay datos en la cola")

        logger.debug("Available devices:")
        if INPUT_DEVICES is not None:
            sd.default.device = INPUT_DEVICES
        logger.debug(sd.query_devices())
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
