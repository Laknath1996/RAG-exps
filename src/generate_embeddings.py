import json
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding_model = "facebook/contriever-msmarco"
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
embedding_model = AutoModel.from_pretrained(embedding_model)

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def divide_to_chapter(text):
    chapter_pattern = r'(?i)^chapter\s+[a-z]+'
    titles = re.findall(chapter_pattern, text, re.MULTILINE)
    splits = re.split(chapter_pattern, text, flags=re.MULTILINE)
    chapters = [f"{title}\n{body.strip()}" for title, body in zip(titles, splits[1:])]
    return chapters

def sentence_chuncking(text, chunk_size=5):
    """
    Split the text into chunks of sentences.
    """
    sentences = sent_tokenize(text)
    chunks = [
        " ".join(sentences[i : i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]
    return chunks

def character_chunking(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for a book and store them in a database.")

    parser.add_argument("--book_path", type=str, default='hp/hp1.txt', help="Path to the book text file.")
    parser.add_argument("--database_path", type=str, default='hp_vdbs', help="Path to the database directory.")
    parser.add_argument("--collection_name", type=str, default='book', help="Name of the collection in the database.")
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of sentences per chunk.")
    parser.add_argument("--device", type=int, default=1, help="Device ID to run the model on (e.g., 0 for GPU 0).")

    args = parser.parse_args()

    book_path = args.book_path
    database_path = args.database_path
    collection_name = args.collection_name
    chunk_size = args.chunk_size
    device = args.device

    # load the book and get the chapters
    text = read_text_file(book_path)
    chapters = divide_to_chapter(text)

    # load the model to device
    embedding_model.to(device)
    embedding_model.eval()

    # configure the database
    client = chromadb.PersistentClient(path=database_path)

    try:
        client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' has been successfully removed.")
    except ValueError as e:
        print(f"Error: {e}")
        print(f"Collection '{collection_name}' may not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    collection = client.get_or_create_collection(collection_name)

    for chapter_id, chapter in enumerate(chapters):
        print(f"Processing chapter {chapter_id + 1}...")

        # remove the chapter title and white spaces
        chapter = re.sub(r"^(CHAPTER \w+\n)(.*\n\n)", "", chapter, flags=re.MULTILINE)
        chapter = chapter.strip()

        # create chunks based on sentences
        chunks = character_chunking(chapter, chunk_size=1000, overlap=200)

        # get embeddings
        inputs = embedding_tokenizer(chunks, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embeddings = mean_pooling(outputs[0], inputs['attention_mask']).cpu().detach().numpy()

        # add to the database
        collection.upsert(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[{"chapter": chapter_id}] * len(chunks),
            ids=[f"ch{chapter_id}_doc{i+1}" for i in range(len(chunks))]
        )