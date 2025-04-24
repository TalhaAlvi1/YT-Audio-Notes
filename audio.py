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
