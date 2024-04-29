import abc
import logging
import re
from typing import Any

import torch
from diffusers import AudioLDM2Pipeline, AutoPipelineForText2Image
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from vertexai.generative_models import GenerativeModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_RATE = 16000


class BaseHint(BaseModel, abc.ABC):
    configs: dict
    hints: list = []
    model: Any = None

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def generate_hint(self, country: str, n_hints: int):
        pass


class TextHint(BaseHint):
    tokenizer: Any = None

    def initialize(self):
        logger.info(
            f"""Initializing text hint with model '{self.configs["model_id"]}'"""
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.configs["model_id"],
            token=self.configs["hf_access_token"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs["model_id"],
            torch_dtype=torch.float16,
            token=self.configs["hf_access_token"],
        ).to(self.configs["device"])
        logger.info("Initialization finisehd")

    def generate_hint(self, country: str, n_hints: int):
        logger.info(f"Generating '{n_hints}' text hints")

        generation_config = GenerationConfig(
            do_sample=True,
            max_new_tokens=self.configs["max_output_tokens"],
            top_k=self.configs["top_k"],
            top_p=self.configs["top_p"],
            temperature=self.configs["temperature"],
        )

        prompt = [
            f'Describe the country "{country}" without mentioning its name\n'
            for _ in range(n_hints)
        ]
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        text_hints = self.model.generate(
            **input_ids.to(self.configs["device"]),
            generation_config=generation_config,
        )

        for idx, text_hint in enumerate(text_hints):
            text_hint = (
                self.tokenizer.decode(text_hint, skip_special_tokens=True)
                .strip()
                .replace(prompt[idx], "")
                .strip()
            )
            text_hint = re.sub(
                re.escape(country), "***", text_hint, flags=re.IGNORECASE
            )

            self.hints.append({"text": text_hint})

        logger.info(f"Text hints '{n_hints}' successfully generated")


class TextHintVertex(BaseHint):
    def initialize(self):
        logger.info(
            f"""Initializing text hint with model '{self.configs["model_id"]}'"""
        )
        self.model = GenerativeModel(self.configs["model_id"])

        logger.info("Initialization finisehd")

    def generate_hint(self, country: str, n_hints: int):
        logger.info(f"Generating '{n_hints}' text hints")

        prompts = [
            f'Describe the country "{country}" without mentioning its name\n'
            for _ in range(n_hints)
        ]

        for prompt in prompts:
            responses = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": self.configs["max_output_tokens"],
                    "top_k": self.configs["top_k"],
                    "top_p": self.configs["top_p"],
                    "temperature": self.configs["temperature"],
                },
            )

            text_hint = responses.candidates[0].content.parts[0].text

            text_hint = re.sub(
                re.escape(country), "***", text_hint, flags=re.IGNORECASE
            )

            self.hints.append({"text": text_hint})

        logger.info(f"Text hints '{n_hints}' successfully generated")


class ImageHint(BaseHint):
    def initialize(self):
        logger.info(
            f"""Initializing image hint with model '{self.configs["model_id"]}'"""
        )
        self.model = AutoPipelineForText2Image.from_pretrained(
            self.configs["model_id"],
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(self.configs["device"])
        logger.info("Initialization finisehd")

    def generate_hint(self, country: str, n_hints: int):
        logger.info(f"Generating '{n_hints}' image hints")
        prompt = [f"An image related to the country {country}" for _ in range(n_hints)]
        img_hints = self.model(
            prompt=prompt,
            num_inference_steps=self.configs["num_inference_steps"],
            guidance_scale=self.configs["guidance_scale"],
        ).images
        self.hints = [{"image": img_hint} for img_hint in img_hints]
        logger.info(f"Image hints '{n_hints}' successfully generated")


class AudioHint(BaseHint):
    def initialize(self):
        logger.info(
            f"""Initializing audio hint with model '{self.configs["model_id"]}'"""
        )
        self.model = AudioLDM2Pipeline.from_pretrained(
            self.configs["model_id"],
            # torch_dtype=torch.float16,  # Not working with MacOS
        ).to(self.configs["device"])
        logger.info("Initialization finisehd")

    def generate_hint(self, country: str, n_hints: int):
        logger.info(f"Generating '{n_hints}' audio hints")
        prompt = f"A sound that resembles the country of {country}"
        negative_prompt = "Low quality"

        audio_hints = self.model(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.configs["num_inference_steps"],
            audio_length_in_s=self.configs["audio_length_in_s"],
            num_waveforms_per_prompt=n_hints,
        ).audios

        for audio_hint in audio_hints:
            self.hints.append(
                {
                    "audio": audio_hint,
                    "sample_rate": SAMPLE_RATE,
                }
            )
        logger.info(f"Audio hints '{n_hints}' successfully generated")
