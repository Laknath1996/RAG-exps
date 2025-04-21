import json
import re
import google.generativeai as genai

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

    Now, generate 15 multiple-choice questions as described above and return your response strictly in JSON format.
    """
    return template.format(history=history, context=context)


def divide_to_chapter(text):
    chapter_pattern = r'(?i)^chapter\s+[a-z]+'
    titles = re.findall(chapter_pattern, text, re.MULTILINE)
    splits = re.split(chapter_pattern, text, flags=re.MULTILINE)
    chapters = [f"{title}\n{body.strip()}" for title, body in zip(titles, splits[1:])]
    return chapters


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Generate questions from a book.")
    parser.add_argument("--path", type=str, default="hp/harry_potter_2.txt", help="Path to the book .txt file.")
    args = parser.parse_args()

    book_path = args.path
    json_path = book_path.replace(".txt", "_question_sets.json")

    # Configure Google Generative AI API
    GOOGLE_API_KEY = "AIzaSyCbk0GN0dnimcDIzfmOK8TQKkbEtiBmG_4"
    genai.configure(api_key=GOOGLE_API_KEY)

    # Load the model
    model = genai.GenerativeModel('gemini-1.5-flash-8b')

    # Load the book
    with open(book_path, 'r', encoding='utf-8') as file:
        text = file.read()
    print("Text successfully read from harry_potter_1.txt")

    # Get individual chapters
    chapters = divide_to_chapter(text)
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

            response = model.generate_content(prompt)

            questions.append(
                {
                    "history_upto": t,
                    "context": k,
                    "output": response.text
                }
            )

            with open(json_path, "w") as f:
                json.dump(questions, f, indent=2)