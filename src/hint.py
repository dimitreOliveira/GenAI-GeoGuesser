import abc
import logging
import os
from typing import Any

import torch
from diffusers import AudioLDM2Pipeline, AutoPipelineForText2Image
from PIL import Image
from pydantic import BaseModel
from scipy.io import wavfile
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseHint(BaseModel):
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
            token=os.environ["HF_ACCESS_TOKEN"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.configs["model_id"],
            torch_dtype=torch.float16,
            token=os.environ["HF_ACCESS_TOKEN"],
        ).to(self.configs["device"])
        logger.info("Initialization finisehd")

    def generate_hint(self, country: str, n_hints: int):
        logger.info(f"Generating '{n_hints}' text hints")
        prompt = [
            f"Describe the country {country} without mentioning its name"
            for _ in range(n_hints)
        ]
        input_ids = self.tokenizer(prompt, return_tensors="pt")
        text_hints = self.model.generate(
            **input_ids.to(self.configs["device"]), max_new_tokens=50
        )
        self.hints = [
            {"text": self.tokenizer.decode(text_hint, skip_special_tokens=True)}
            for text_hint in text_hints
        ]


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
            num_inference_steps=1,
            guidance_scale=0.0,
        ).images
        self.hints = [{"image": img_hint} for img_hint in img_hints]


class AudioHint(BaseHint):
    def initialize(self):
        logger.info(
            f"""Initializing audio hint with model '{self.configs["model_id"]}'"""
        )
        self.model = AudioLDM2Pipeline.from_pretrained(
            self.configs["model_id"],
            torch_dtype=torch.float16,
        ).to(self.configs["device"])
        logger.info("Initialization finisehd")

    def generate_hint(self, country: str, n_hints: int):
        logger.info(f"Generating '{n_hints}' audio hints")
        prompt = f"A sound that resembles the country of {country}"
        negative_prompt = "Low quality."

        audio_hints = self.model(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=200,
            audio_length_in_s=10.0,
            # num_inference_steps=20,
            # audio_length_in_s=2,
            num_waveforms_per_prompt=n_hints,
        ).audios

        hints = []
        for audio_hint in audio_hints:
            hints.append(
                {
                    "audio": audio_hint,
                    "sample_rate": 16000,
                }
            )
        self.hints = hints
