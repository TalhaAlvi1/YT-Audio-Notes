import os
import argparse
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pytube
import whisper
import torch
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YTAudioNotes:
    """Main class for handling YouTube audio extraction, transcription, and summarization."""

    def __init__(self, use_openai_api: bool = False, openai_api_key: Optional[str] = None,
                 whisper_model: str = "base", language: Optional[str] = None):
        """
        Initialize the YT-Audio-Notes tool.

        Args:
            use_openai_api: Whether to use OpenAI's API for transcription
            openai_api_key: OpenAI API key (required if use_openai_api is True)
            whisper_model: Whisper model to use ("tiny", "base", "small", "medium", "large")
            language: Language code for transcription (optional)
        """
        self.use_openai_api = use_openai_api
        self.openai_api_key = openai_api_key
        self.whisper_model = whisper_model
        self.language = language

        if self.use_openai_api and not self.openai_api_key:
            logger.warning("OpenAI API key not provided. Falling back to local Whisper model.")
            self.use_openai_api = False

        if not self.use_openai_api:
            # Load local Whisper model
            logger.info(f"Loading Whisper model: {whisper_model}")
            self.model = whisper.load_model(whisper_model)

        # Initialize the summarization pipeline
        logger.info("Initializing summarization pipeline")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
