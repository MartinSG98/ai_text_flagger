import json
import random
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # replace multiple spaces/newlines with single space
    return text.strip()

