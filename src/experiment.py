# TODO: Change the variable names such as history_upto, context_chapter, etc. to be more meaningful

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import json
import re
import os

import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

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
    
def get_prompt(context, question, a, b, c, d, past_context=None):
    if past_context is None:
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
    else:
        prompt = f"""
        You are given a current context, a retrieved past context and a multiple-choice question. Choose the **best answer** from the options (A, B, C, D). Respond with only the **letter corresponding to the correct answer**.

        Context:
        {context}

        Past Context:
        {past_context}

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

def get_rag_query(question, a, b, c, d):
    query = f"""
    Question:
    {question}

    Options:
    A. {a}
    B. {b}
    C. {c}
    D. {d}
    """
    return query

def get_model_answer(prompt, model, tokenizer):
    model_input = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids_length = model_input['input_ids'].shape[1]

    with torch.no_grad():
        model_output = model.generate(**model_input, max_new_tokens=1)
    
    response = tokenizer.decode(model_output[0][input_ids_length:], skip_special_tokens=True)
    return response

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run the experiment script with specified parameters.")

    # data args
    parser.add_argument('--chapters_path', type=str, default='hp/hp1_2_chapters.json', help='Path to the book text file.')
    parser.add_argument('--questions_path', type=str, default='hp/hp1_2_questions.json', help='Path to the JSON file containing question sets.')
    
    # rag args
    parser.add_argument('--database_path', type=str, default='hp_vdbs/hp', help='Path to the RAG database.')
    parser.add_argument('--collection_name', type=str, default='book1_2', help='Name of the RAG collection.')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top documents to retrieve with RAG.')
    
    # experiment args
    parser.add_argument('--use_adapters', action='store_true', help='Flag to use LoRA adapters.')
    parser.add_argument('--use_rag', action='store_true', help='Flag to use RAG for retrieval.')
    
    # other args
    parser.add_argument('--results_file', type=str, default='results/results.json', help='Path to save the results JSON file.')
    parser.add_argument('--device', type=int, default=1, help='Device ID to use for the model.')

    args = parser.parse_args()

    chapters_path = args.chapters_path
    questions_path = args.questions_path
    database_path = args.database_path
    collection_name = args.collection_name
    use_adapters = args.use_adapters
    use_rag = args.use_rag
    top_k = args.top_k
    results_file = args.results_file
    device = args.device

    # get whole text, chapters, and questions
    book_name = os.path.basename(chapters_path).split('.')[0].replace('_chapters', '')
    with open(chapters_path) as f:
        chapters = json.load(f)
    question_sets = read_list_from_json(questions_path)

    # load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if use_rag:
        from rag import RAG
        rag = RAG(database_path=database_path, collection_name=collection_name)

    results = []

    for question_set in question_sets:

        current = question_set['history_upto']      # current chapter_id (history = 0...chapter_id)
        context_chapter = question_set['context']   # future chapter_id that will be used as context
        context = chapters[context_chapter]         # context (= chapter content)
        q_list = question_set['output']             # set of questions based on the context and history

        if use_adapters:
            print(f"Loading LoRA adapter for chapter {current}...")
            model = PeftModel.from_pretrained(base_model, f"lora_adapters/{book_name}_ch{current}")
        else:
            if use_rag:
                print(f"Loading base model + RAG for chapter {current}...")
            else:
                print(f"Loading base model for chapter {current}...")
            model = base_model
        
        model.to(device)
        model.eval()

        truths = []
        preds = []

        for q in q_list:

            question = q['question']
            true_answer = q['answer']
            a, b, c, d = q['A'], q['B'], q['C'], q['D']

            if use_rag:
                past_context = rag.retrieve(
                    get_rag_query(question, a, b, c, d), 
                    scope=[i for i in range(current+1)], 
                    top_k=top_k
                )
                past_context = "\n-----\n".join(past_context)
                prompt = get_prompt(context, question, a, b, c, d, past_context)
            else:
                prompt = get_prompt(context, question, a, b, c, d)
            
            model_answer = get_model_answer(prompt, model, tokenizer).strip()

            print(f"t = {current}, k = {context_chapter}, True answer: {true_answer}, Model answer: {model_answer}")

            truths.append(true_answer)
            preds.append(model_answer)

        error = np.mean(np.array(truths) != np.array(preds))
        
        results.append(
            {
                't': current,
                'k': context_chapter,
                'error': error,
                'truths': truths,
                'preds': preds
            }
        )

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
