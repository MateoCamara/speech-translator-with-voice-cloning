import json
import time
import logging
import torch

# import wandb
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

        # Check if the input is the flush command
        if new_input == "__FLUSH__":
            if self.incomplete_buffer.strip():
                logger.info("Flush command received with non-empty buffer. Forcing translation.")
                language_code = "es"
                translated_text = self.translate_text(self.incomplete_buffer)
                self.update_history(self.incomplete_buffer)
                # Clear the buffer and reset counter
                self.incomplete_buffer = ""
                self.incomplete_count = 0
                yield (translated_text, language_code)
            else:
                logger.info("Flush command received but buffer is empty. Ending pipeline.")
                yield None
            return

        # Normal processing: combine new input with the incomplete buffer.
        combined_text = (self.incomplete_buffer + " " + new_input).strip()

        # Validate the combined text.
        complete_sentences, remaining_text = self.check_sentence_completion(combined_text)

        if complete_sentences:
            # Reset counter and update history if complete sentence(s) are detected.
            self.incomplete_count = 0
            self.update_history(complete_sentences)
            # Set the buffer to the remaining incomplete text.
            self.incomplete_buffer = remaining_text.strip()
            language_code = "es"
            translated_text = self.translate_text(complete_sentences)
            yield (translated_text, language_code)
        else:
            # No complete sentence detected: accumulate the text and increment the counter.
            self.incomplete_buffer = combined_text
            self.incomplete_count += 1
            logger.info("No complete sentence detected. Incomplete count: %d", self.incomplete_count)
            # If 5 iterations have passed, force translation.
            if self.incomplete_count >= 5:
                logger.info("Buffer limit reached. Forcing translation of accumulated text.")
                language_code = "es"
                translated_text = self.translate_text(self.incomplete_buffer)
                self.update_history(self.incomplete_buffer)
                self.incomplete_buffer = ""
                self.incomplete_count = 0
                yield (translated_text, language_code)
            else:
                yield None

    def update_history(self, sentence):
        """
        Updates the complete sentence history (stores only the last 5 entries).
        """
        self.complete_history.append(sentence)
        if len(self.complete_history) > 5:
            self.complete_history = self.complete_history[-5:]
        logger.debug("Updated complete history: %s", self.complete_history)

    def check_sentence_completion(self, text):
        """
        Validates the given text and extracts complete sentence(s) from the beginning.

        IMPORTANT:
         - The prompt includes a "Context" section with previous complete sentences,
           but this context is provided solely for background understanding.
         - The LLM must process ONLY the provided input text (i.e. the combination of the incomplete buffer and new input).
         - The output JSON must contain two keys:
             "complete_sentences": string with the complete sentences extracted (empty if none).
             "incomplete_text": string with the remaining text that does not form a complete sentence.
         - Do NOT include any text from the context in either output field.
         - The text comes from an automated speech recognition system and may have errors.
         - Ensure that if only a partial sentence is present at the beginning, it is left in "incomplete_text".

        Example:
          Text: "The cat were. Alone looking at the fire. Then the owner."
          Expected output:
            {
                "complete_sentences": "The cat were alone looking at the fire.",
                "incomplete_text": "Then the owner."
            }

        Edge-cases considered:
          - If the entire input is incomplete, return empty "complete_sentences" and the full input in "incomplete_text".
          - Do not let context data be confundido con el input a procesar.
        """
        # Build background context (only for understanding, NOT to be processed)
        context = ""
        if self.complete_history:
            context = "Context (for background only, do not include in output): " + " | ".join(
                self.complete_history) + "\n"

        # The prompt clearly separates context from the input text
        validation_prompt = f"""
        Context (for background only, do not include in output): {" | ".join(self.complete_history) if self.complete_history else ""}

        You will receive an input text that represents a temporal sequence of utterances. This text is a combination of previously accumulated incomplete speech and new speech from an automated speech recognition system. Your task is to extract contiguous complete sentence(s) from the very beginning of the input text, ensuring strict temporal coherence.

        IMPORTANT:
        - Process ONLY the input text provided after this context.
        - Extract complete sentence(s) that appear continuously from the start of the input text.
        - Assume that if the input ends with a complete sentence, then any previously incomplete fragment (accumulated earlier) has now been completed by the speaker and should be treated as complete.
        - Return a JSON object with exactly two keys:
            "complete_sentences": A string containing all contiguous complete sentence(s) extracted from the beginning of the input text.
            "incomplete_text": A string containing the remaining part of the input text that does not form a complete sentence.
        - Do NOT include any context information in your outputs.
        - The input text may contain recognition errors; you may make minimal corrections for clarity, but do not add or remove content beyond what is necessary.
        - If the entire input text is incomplete, return an empty string for "complete_sentences" and the full input text for "incomplete_text".

        Example:
        Input text: "The cat were. Alone looking at the fire. Then the owner."
        Expected output:
        {{
            "complete_sentences": "The cat were alone looking at the fire.",
            "incomplete_text": "Then the owner."
        }}

        Here is the input text to process: '{text}'
        """

        logger.debug("Validation prompt: %s", validation_prompt)

        try:
            start_time = time.time()
            response = self.client_openai.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': validation_prompt}],
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
            logger.debug("Validation response: %s", response_content)
            response_json = json.loads(response_content)
            complete_sentences = response_json.get('complete_sentences', '').strip()
            incomplete_text = response_json.get('incomplete_text', '').strip()
            end_time = time.time()
            latency = end_time - start_time
            # if DEBUG_LOGGING:
            #     wandb.log({"LLM_validation_latency": latency})
            return complete_sentences, incomplete_text
        except Exception as e:
            logger.error("Error during sentence validation: %s", str(e))
            # On error, assume no complete sentence was extracted
            return "", text

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
                # wandb.log({"LLM_translation_latency": latency})

                with open("./tests/latency/translation_with_uncompleted_sentences.txt", "a", encoding="utf-8") as archivo:
                    archivo.write(" " + translated_text)


            return translated_text
        except Exception as e:
            logger.error("Error during translation: %s", str(e))
            return ""
