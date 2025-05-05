# TODO: Add a condition to check whether the keys are correct
# keys = ['question', 'A', 'B', 'C', 'D', 'answer', 'explanation']
# all([list(q.keys()) == keys for q in q_list]) should be true

import json
import re
import google.generativeai as genai

KEYS = ['question', 'A', 'B', 'C', 'D', 'answer', 'explanation']

def get_prompt(history, context):
    template = """
    You are reading a book. The portion you have already read is referred to as the **history**, and a new **context** (an excerpt from a later part of the book) is provided. 

    Your task is to generate **10 multiple-choice questions** based on the *context*. However, the **correct answers to these questions must depend on the information provided in the history**—not just the context alone.

    Each question should have:
    - A clear and concise question stem
    - Four answer choices (A, B, C, D)
    - One correct answer that relies on the history
    - A brief explanation of why the correct answer is correct (optional but recommended for clarity)

    Here are the inputs:

    **History**:  
    {history}

    **Context**:  
    {context}

    Now, generate 10 multiple-choice questions as described above. Return your response strictly in the following JSON format.

    ```json
    [
        {{
            "question": "Your question here?",
            "A": "Option A text",
            "B": "Option B text",
            "C": "Option C text",
            "D": "Option D text"
            "answer": "C",
            "explanation": "Explain why C is correct based on the history."
        }},
    ...
    ]
    ```
    """
    return template.format(history=history, context=context)


def divide_to_chapter(text):
    chapter_pattern = r'(?i)^chapter\s+[a-z]+'
    titles = re.findall(chapter_pattern, text, re.MULTILINE)
    splits = re.split(chapter_pattern, text, flags=re.MULTILINE)
    chapters = [f"{title}\n{body.strip()}" for title, body in zip(titles, splits[1:])]
    return chapters

def preprocess_text(text):
    text = re.sub(r"^(CHAPTER \w+\n)(.*\n\n)", "", text, flags=re.MULTILINE)
    text = text.strip()
    return text

def extract_questions(generated_str):
    match = re.search(r'\[.*?\]', generated_str, re.DOTALL)
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error decoding JSON: {e}")
        return None

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Generate questions from a book.")
    parser.add_argument("--chapters_path", type=str, default="hp/hp1_2_chapters.json", help="Path to json file containing book chapters.")
    parser.add_argument("--questions_path", type=str, default="hp/hp1_2_questions.json", help="Path to json file that will contain the questions.")
    args = parser.parse_args()

    chapters_path = args.chapters_path
    questions_path = args.questions_path

    # Configure Google Generative AI API
    GOOGLE_API_KEY = "AIzaSyCbk0GN0dnimcDIzfmOK8TQKkbEtiBmG_4"
    genai.configure(api_key=GOOGLE_API_KEY)

    # Load the model
    model = genai.GenerativeModel('gemini-1.5-flash-8b')

    # Get individual chapters
    with open(chapters_path) as f:
        chapters = json.load(f)
    T = len(chapters)
    print(f"Number of chapters: {T}")

    # Generate questions
    questions = []
    for t in range(T-1):
        history = "\n\n\n\n\n".join(chapters[:t+1])
        for k in range(t+1, T):
            context = chapters[k]

            # get prompt
            prompt = get_prompt(history, context)

            print(f"t={t}, k={k}, total tokens : {model.count_tokens(prompt)}")

            success = False
            while not success:
                response = model.generate_content(prompt)
                q_list = extract_questions(response.text)
                if q_list is not None and len(q_list) == 10:
                    if all([list(q.keys()) == KEYS for q in q_list]):
                        success = True
                        print("Successfully generated 10 questions.")

            questions.append(
                {
                    "history_upto": t,
                    "context": k,
                    "output": q_list
                }
            )

            with open(questions_path, "w") as f:
                json.dump(questions, f, indent=4)