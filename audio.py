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

    def generate_notes(self, transcript: str, max_length: int = 500,
                       min_length: int = 150) -> str:
        """
        Generate summarized notes from a transcript.

        Args:
            transcript: The transcript text
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary

        Returns:
            Summarized notes in bullet-point format
        """
        logger.info("Generating notes from transcript")

        try:
            # Split long transcripts into chunks for processing
            chunks = split_text(transcript, max_chunk_size=1000)
            summaries = []

            for i, chunk in enumerate(chunks):
                logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")

                # Get summary for this chunk
                summary = self.summarizer(
                    chunk,
                    max_length=max_length // len(chunks),
                    min_length=min_length // len(chunks),
                    do_sample=False
                )[0]["summary_text"]

                summaries.append(summary)

            # Combine summaries
            combined_summary = " ".join(summaries)

            # Convert to bullet points
            bullet_points = self._convert_to_bullet_points(combined_summary)

            logger.info("Notes generation completed")
            return bullet_points

        except Exception as e:
            logger.error(f"Error generating notes: {str(e)}")
            raise

    def _convert_to_bullet_points(self, text: str) -> str:
        """Convert a text into bullet points."""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Create bullet points
        bullet_points = ""
        for sentence in sentences:
            if sentence.strip():
                bullet_points += f"* {sentence.strip()}\n"

        return bullet_points

    def save_to_file(self, content: str, output_file: str) -> None:
        """Save content to a file."""
        logger.info(f"Saving content to: {output_file}")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Content saved to: {output_file}")

        except Exception as e:
            logger.error(f"Error saving to file: {str(e)}")
            raise
    def process_video(self, url: str, output_dir: str = "output",
                     include_timestamps: bool = False) -> Tuple[str, str]:
        """
        Process a YouTube video: download, transcribe, and generate notes.

        Args:
            url: YouTube video URL
            output_dir: Directory to save output files
            include_timestamps: Whether to include timestamps in the transcript

        Returns:
            Tuple of (transcript_file_path, notes_file_path)
        """
        logger.info(f"Processing video: {url}")

        try:
            # Download audio
            audio_file = self.download_audio(url, output_dir)

            # Get video title for naming
            # Extract the filename without extension
            base_name = os.path.basename(audio_file)
            video_title = os.path.splitext(base_name)[0]
            safe_title = "".join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in video_title)
            safe_title = safe_title.replace(' ', '_')

            # Transcribe audio
            transcription_result = self.transcribe_audio(audio_file, include_timestamps)
            transcript = transcription_result["formatted_transcript"]

            # Generate notes
            notes = self.generate_notes(transcript)

            # Save transcript
            transcript_file = os.path.join(output_dir, f"{safe_title}_transcript.txt")
            self.save_to_file(transcript, transcript_file)

            # Save notes
            notes_file = os.path.join(output_dir, f"{safe_title}_notes.md")
            self.save_to_file(notes, notes_file)

            logger.info("Video processing completed")
            return transcript_file, notes_file

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def split_text(text: str, max_chunk_size: int = 1000) -> list:
    """Split text into chunks of maximum size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        if current_size + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for the space

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
# Run the tool - only asking for YouTube URL
def run_yt_audio_notes():
    print("🎧 YT-Audio-Notes: YouTube to Audio Transcription and Notes 📝")
    print("===========================================================")

    # Only ask for YouTube URL, use defaults for everything else
    url = input("Enter YouTube URL: ")

    print("\nProcessing your request with default settings:")
    print("- Output directory: output")
    print("- Whisper model: base")
    print("- Include timestamps: No")
    print("- Transcript format: txt")
    print("- Notes format: md")
    print("\nThis may take some time depending on the video length...\n")
