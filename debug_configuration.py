DEBUG_LOGGING = False  # Activates logging in wandb
FLUSH_VAD = 10 # In seconds. Limit to the continuous speech. If no silence detected before this N seconds it'll proceed in the pipeline
USE_COMPLETION_LLM = False # Handles using the extra LLM to try to translate only complete sentences
INPUT_DEVICES = None # Or a tuple with (input, output). See readme!
OUTPUT_DEVICES = None # Or a tuple with (input, output). See readme!
EUROPARL_PATH = None # Path to the europarl dataset, for offline testing.