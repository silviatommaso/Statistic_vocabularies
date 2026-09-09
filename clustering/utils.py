import re

def normalize(text):
    text = text.strip().strip('"').strip("'")
    text = re.sub(r'^[-–—\s]+', '', text)
    text = re.sub(r'[()]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def word_tokens(text):
    return re.findall(r"[A-Za-z0-9]+", text.lower())