import json
import time
import logging
import torch

import wandb
from openai import OpenAI
from baseHandler import BaseHandler
from rich.console import Console

from debug_configuration import DEBUG_LOGGING

logger = logging.getLogger(__name__)
console = Console()

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
}


class LanguageModelHandler(BaseHandler):
    """
    Handles tasks for sentence validation and translation.
    """

    def setup(self,
              model_name="microsoft/Phi-3-mini-4k-instruct",
              device="cuda",
              torch_dtype="float16",
              gen_kwargs={},
              user_role="user",
              chat_size=1,
              init_chat_role=None,
              init_chat_prompt="You are a helpful AI assistant."):
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.user_role = user_role
        self.client_openai = OpenAI()
        self.start_time = time.time()
        # Buffer to accumulate incomplete text from previous iterations
        self.incomplete_buffer = ""
        # Counter for consecutive iterations without complete sentence
        self.incomplete_count = 0
        # History of last 5 complete sentences for context (background only)
        self.complete_history = []


    def process(self, prompt):
        """
        Processes input text:
          - If the input is the flush command "__FLUSH__":
              * If the incomplete buffer contains text, force its translation.
              * Otherwise, end the pipeline without output.
          - Otherwise, combine the incomplete buffer with the new input, validate, and either translate or accumulate.
        """
        new_input = prompt[0].strip()
        logger.debug("Received new input: %s", new_input)

        language_code = "es"
        translated_text = self.translate_text(new_input)
        self.update_history(translated_text)
        yield (translated_text, language_code)


    def update_history(self, sentence):
        """
        Updates the complete sentence history (stores only the last 5 entries).
        """
        self.complete_history.append(sentence)
        if len(self.complete_history) > 5:
            self.complete_history = self.complete_history[-5:]
        logger.debug("Updated complete history: %s", self.complete_history)

    def translate_text(self, text_to_translate):
        """
        Translates the given text to Spanish.

        IMPORTANT:
         - The background context (previous complete sentences) is provided only as information to help with coherence.
         - Do NOT include the context text in the actual text to translate.
         - Follow the rules:
              * If the text is already in Spanish, return it as is.
              * Translate any numbers or special characters into words (e.g., "4x4" → "four by four").
              * Spell out acronyms (e.g., "USA" → "U S A").
              * Consider that pronunciation errors may occur (e.g., "yawn" may sound like "yarn" or "young").
         - If the text is incomplete, translate as accurately as possible without adding extra content.

        Edge-cases considered:
         - Ensure that context is only used to understand the subject and is not part of the translation.
        """
        # Build background context (only for understanding)
        context = ""
        if self.complete_history:
            context = "Context (for background only): " + " | ".join(self.complete_history) + "\n"

        translation_prompt = (
            f"""
            {context}
            You are an expert translator from English to Spanish.
            IMPORTANT:
            - Translate ONLY the text provided after this instruction.
            - Do NOT include any of the context text in your translation.
            - Do not add any commentary or extra text; output only the translation.
            - If the text is already in Spanish, return it as is.
            - Convert digits and special characters to words (e.g., "4x4" → "four by four", "3D" → "three dimensional").
            - Spell out acronyms (e.g., "USA" → "U S A").
            - Handle any pronunciation errors appropriately without adding information.

            Here is the text in English to translate: '{text_to_translate}'

            Translate it to Spanish:
            """
        )
        logger.debug("Translation prompt: %s", translation_prompt)
        try:
            start_time = time.time()
            response = self.client_openai.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': translation_prompt}]
            )
            translated_text = response.choices[0].message.content.strip()
            logger.debug("Translated text: %s", translated_text)
            end_time = time.time()
            latency = end_time - start_time
            if DEBUG_LOGGING:
                wandb.log({"LLM_translation_latency": latency})

                with open("./tests/latency/translation.txt", "a", encoding="utf-8") as archivo:
                    archivo.write(" " + translated_text)


            return translated_text
        except Exception as e:
            logger.error("Error during translation: %s", str(e))
            return ""
