import re

def preprocess_text(text):
    text = re.sub(r"^(CHAPTER \w+\n)(.*\n\n)", "", text, flags=re.MULTILINE)
    text = text.strip()
    return text

def divide_to_chapter(text):
    chapter_pattern = r'(?i)^chapter\s+[a-z]+'
    titles = re.findall(chapter_pattern, text, re.MULTILINE)
    splits = re.split(chapter_pattern, text, flags=re.MULTILINE)
    chapters = [f"{title}\n{body.strip()}" for title, body in zip(titles, splits[1:])]
    return chapters