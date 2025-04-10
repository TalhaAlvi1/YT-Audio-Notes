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

    def download_audio(self, url: str, output_dir: str = "output") -> str:
        """
        Download audio from a YouTube video URL.

        Args:
            url: YouTube video URL
            output_dir: Directory to save the audio file

        Returns:
            Path to the downloaded audio file
        """
        logger.info(f"Downloading audio from: {url}")
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Download the YouTube video
            yt = pytube.YouTube(url)
            video_title = yt.title
            logger.info(f"Video title: {video_title}")

            # Extract audio
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_file = audio_stream.download(output_path=output_dir)

            # Rename to mp3 (optional, Whisper can handle various formats)
            base, _ = os.path.splitext(audio_file)
            mp3_file = f"{base}.mp3"
            os.rename(audio_file, mp3_file)

            logger.info(f"Audio saved to: {mp3_file}")
            return mp3_file

        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            raise

    def transcribe_audio(self, audio_file: str, include_timestamps: bool = False) -> Dict[str, Any]:
        """
        Transcribe the audio file.

        Args:
            audio_file: Path to the audio file
            include_timestamps: Whether to include timestamps in the transcript

        Returns:
            Dictionary containing the transcript and metadata
        """
        logger.info(f"Transcribing audio file: {audio_file}")

        try:
            if self.use_openai_api:
                import openai
                openai.api_key = self.openai_api_key

                with open(audio_file, "rb") as audio:
                    response = openai.Audio.transcribe(
                        model="whisper-1",
                        file=audio,
                        language=self.language
                    )
                result = response.to_dict()
            else:
                # Use local Whisper model
                transcribe_options = {}
                if self.language:
                    transcribe_options["language"] = self.language

                result = self.model.transcribe(
                    audio_file,
                    **transcribe_options
                )

            # Extract and format transcript
            transcript = result["text"]

            # Include timestamps if requested
            if include_timestamps and not self.use_openai_api:
                segments = result["segments"]
                transcript_with_timestamps = ""

                for segment in segments:
                    start_time = format_timestamp(segment["start"])
                    text = segment["text"]
                    transcript_with_timestamps += f"[{start_time}] {text}\n"

                result["formatted_transcript"] = transcript_with_timestamps
            else:
                result["formatted_transcript"] = transcript

            logger.info("Transcription completed")
            return result

        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            raise
