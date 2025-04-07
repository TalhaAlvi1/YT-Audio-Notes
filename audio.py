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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
