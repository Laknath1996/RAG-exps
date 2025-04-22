# TODO: Implement RAG baseline

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import json
import re

import numpy as np


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16)
model.to(1)


def divide_to_chapter(text):
    chapter_pattern = r'(?i)^chapter\s+[a-z]+'
    titles = re.findall(chapter_pattern, text, re.MULTILINE)
    splits = re.split(chapter_pattern, text, flags=re.MULTILINE)
    chapters = [f"{title}\n{body.strip()}" for title, body in zip(titles, splits[1:])]
    return chapters

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def read_list_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    else:
        raise ValueError("The JSON file does not contain a list.")
    
def get_prompt(context, question, a, b, c, d):
    prompt = f"""
    You are given a context and a multiple-choice question. Choose the **best answer** from the options (A, B, C, D). Respond with only the **letter corresponding to the correct answer**.

    Context:
    {context}

    Question:
    {question}

    Options:
    A. {a}
    B. {b}
    C. {c}
    D. {d}

    Answer:
    """
    return prompt

def get_model_answer(prompt):
    model_input = tokenizer(prompt, return_tensors="pt").to(1)
    input_ids_length = model_input['input_ids'].shape[1]
    model_output = model.generate(**model_input, max_new_tokens=1)
    response = tokenizer.decode(model_output[0][input_ids_length:], skip_special_tokens=True)
    return response

if __name__ == "__main__":

    book_path = 'hp/harry_potter_1.txt'
    questions_path = 'hp/harry_potter_1_question_sets.json'
    text = read_text_file(book_path)
    chapters = divide_to_chapter(text)
    question_sets = read_list_from_json(questions_path)

    results = []

    for question_set in question_sets:
        history_upto = question_set['history_upto']
        context_chapter = question_set['context']
        context = chapters[context_chapter]
        q_list = question_set['output']

        truths = []
        preds = []
        error = None

        if q_list is not None:
            for q in q_list:

                question = q['question']
                true_answer = q['answer']
                a, b, c, d = q['A'], q['B'], q['C'], q['D']

                prompt = get_prompt(context, question, a, b, c, d)
                
                model_answer = get_model_answer(prompt).strip()

                print(f"t = {history_upto}, k = {context_chapter}, True answer: {true_answer}, Model answer: {model_answer}")

                truths.append(true_answer)
                preds.append(model_answer)

            error = np.mean(np.array(truths) != np.array(preds))
        
        results.append(
            {
                't': history_upto,
                'k': context_chapter,
                'error': error,
                'truths': truths,
                'preds': preds
            }
        )


    filename = 'results/results.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
