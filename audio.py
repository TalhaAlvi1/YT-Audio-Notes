!pip install pytube==12.1.0 openai-whisper torch transformers sentencepiece protobuf openai yt-dlp

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import re

# Use yt-dlp instead of pytube to avoid HTTP 400 errors
import yt_dlp
import whisper
import torch
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
class YTAudioNotes:
    """Main class for handling YouTube audio extraction, transcription, and summarization."""

    def __init__(self, whisper_model: str = "base", language: Optional[str] = None):
        """
        Initialize the YT-Audio-Notes tool.

        Args:
            whisper_model: Whisper model to use ("tiny", "base", "small", "medium", "large")
            language: Language code for transcription (optional)
        """
        self.whisper_model = whisper_model
        self.language = language

        # Load local Whisper model
        logger.info(f"Loading Whisper model: {whisper_model}")
        self.model = whisper.load_model(whisper_model)

        # Initialize the summarization pipeline
        logger.info("Initializing summarization pipeline")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def download_audio(self, url: str, output_dir: str = "output") -> str:
        """
        Download audio from a YouTube video URL using yt-dlp.

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

            # Set options for downloading
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': False,
                'no_warnings': False
            }

            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info['title']

            # Get the path of the downloaded file
            mp3_file = os.path.join(output_dir, f"{video_title}.mp3")
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
            if include_timestamps:
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
