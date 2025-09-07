import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import json
import re
import os

from rich.live import Live
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn, TimeElapsedColumn

import numpy as np

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def read_list_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    else:
        raise ValueError("The JSON file does not contain a list.")
    
def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
def get_prompt(question, a, b, c, d, context=None):
    if context:
        prompt = f"""
        You are given a multiple-choice question. Additionally, you are given some background information within <context>. Choose the best answer from the options (A, B, C, D) based on the <context>. Respond with only the letter corresponding to the correct answer.

        <context>
        {context}
        </context>

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
        You are given a multiple-choice question. Choose the best answer from the options (A, B, C, D). Respond with only the letter corresponding to the correct answer.

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

def get_model_answer(prompt, model, tokenizer):
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1,
        num_return_sequences=5
    )

    generated_ids = [
        output_ids[model_inputs.input_ids.shape[1]:] for output_ids in generated_ids
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    response = [r.strip() for r in response]
    return response

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run the experiment script with specified parameters.")

    # data args
    parser.add_argument('--chapters_path', type=str, default='hp/hp1_3_chapters.json', help='Path to the book text file.')
    parser.add_argument('--questions_path', type=str, default='hp/hp1_3_chapterwise_questions.json', help='Path to the JSON file containing question sets.')
    
    # rag args
    parser.add_argument('--database_path', type=str, default='hp_vdbs/hp', help='Path to the RAG database.')
    parser.add_argument('--collection_name', type=str, default='book1_3', help='Name of the RAG collection.')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top documents to retrieve with RAG.')
    
    # LoRA adapter args
    parser.add_argument('--use_adapters', action='store_true', help='Flag to use LoRA adapters.')
    parser.add_argument('--ckpt_id', type=str, default='None', help='Checkpoint ID for the LoRA adapter.')
    
    # RAG args
    parser.add_argument('--use_rag', action='store_true', help='Flag to use RAG for retrieval.')
    parser.add_argument('--rag_memory', type=int, default=-1, help='Memory size (in terms of chapters) for RAG. Default is -1 (no limit).')
    
    # other args
    parser.add_argument('--load_unlearned_model', action='store_true', help='Flag to load the unlearned model instead of the learned one.')
    parser.add_argument('--unlearned_ckpt_id', type=int, default=5, help='Unlearned model checkpoint.')
    parser.add_argument('--results_file', type=str, default='results/results.json', help='Path to save the results JSON file.')
    parser.add_argument('--device', type=int, default=1, help='Device ID to use for the model.')


    args = parser.parse_args()

    chapters_path = args.chapters_path
    questions_path = args.questions_path
    database_path = args.database_path
    collection_name = args.collection_name
    use_adapters = args.use_adapters
    ckpt_id = args.ckpt_id
    use_rag = args.use_rag
    rag_memory = args.rag_memory
    top_k = args.top_k
    results_file = args.results_file
    device = args.device
    load_unlearned_model = args.load_unlearned_model
    unlearned_ckpt_id = args.unlearned_ckpt_id

    question_sets = read_list_from_json(questions_path)

    # load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if use_rag:
        from rag import RAG
        rag = RAG(database_path=database_path, collection_name=collection_name)

        def fetch_context(query, scope):
            return rag.retrieve(query, scope=scope, top_k=top_k)

    results = []
    num_chapters = len(question_sets)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[green]{task.completed}[/]/[bold]{task.total}"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    task = progress.add_task("[green]Processing chapters...", total=num_chapters)

    with Live(progress, refresh_per_second=4):
        for id in range(num_chapters):
            question_set = question_sets[:id+1]

            q_list, q_id = [], []
            for i, qs in enumerate(question_set):
                q_list.extend(qs['questions'])
                q_id.extend([i]*len(qs['questions']))
        
            truths = []
            preds = []

            if use_adapters:
                model.to('cpu')
                del model
                torch.cuda.empty_cache()

                print(f"Loading adapters for chapter {id}...")
                base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
                lora_adapter_path = f'lora_adapters/hp1_2_new/hp1_2_ch{id}'
                model = PeftModel.from_pretrained(base_model, lora_adapter_path)
                model.to(device)

            for q in q_list:

                question = q['question']
                true_answer = q['answer']
                a, b, c, d = q['A'], q['B'], q['C'], q['D']

                context = fetch_context(question, scope=list(range(id+1))) if use_rag else None

                prompt = get_prompt(question, a, b, c, d, context=context)
                
                model_answer = get_model_answer(prompt, model, tokenizer)

                # print(f"current chapter = {id}, True answer: {true_answer}, Model answer: {model_answer}")

                truths.append([true_answer]*5)
                preds.append(model_answer)
            
            results.append(
                {
                    'chapter': id,
                    'truths': truths,
                    'preds': preds,
                    'question_ids': q_id
                }
            )

            progress.update(task, advance=1)

    save_json(results, results_file)
