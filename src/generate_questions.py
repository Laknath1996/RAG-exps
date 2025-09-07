import json
import re
import google.generativeai as genai
from google.api_core import retry, exceptions
import time

KEYS = ['question', 'A', 'B', 'C', 'D', 'answer', 'explanation']

def get_prompt(history, num_questions=10):
    template = """
    You are reading a book. The portion you have already read is referred to as the <history>. 

    Your task is to generate {num_questions} multiple-choice questions based on the <history>. The correct answers to these questions must depend only on the information provided in the <history>.

    Each question should have:
    - A clear and concise question stem
    - Four answer choices (A, B, C, D)
    - One correct answer that relies on the history
    - A brief explanation of why the correct answer is correct (optional but recommended for clarity)

    Here is the history:

    <history>
    {history}
    </history>
    
    Now, generate {num_questions} multiple-choice questions as described above. Return your response strictly in the following JSON format.

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
    return template.format(history=history, num_questions=num_questions)


def parse_questions(generated_str):
    match = re.search(r'\[.*?\]', generated_str, re.DOTALL)
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error decoding JSON: {e}")
        return None

def is_retryable_exception(e):
    # Retry on 429 (Too Many Requests) and 5xx (Server Errors)
    if isinstance(e, exceptions.ResourceExhausted) or (
        isinstance(e, exceptions.ServiceUnavailable) or isinstance(e, exceptions.InternalServerError)
    ):
        print(f"Encountered retryable error. Retrying...")
        return True
    return False

@retry.Retry(
    predicate=is_retryable_exception,
    initial=10.0,
    maximum=10.0,
)
def generate_text_with_retry(model, prompt):
    response = model.generate_content(prompt)
    return response

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Generate questions from a book.")
    parser.add_argument("--chapters_path", type=str, default="hp/hp1_2_chapters.json", help="Path to json file containing book chapters.")
    parser.add_argument("--questions_path", type=str, default="hp/hp1_2_questions.json", help="Path to json file that will contain the questions.")
    parser.add_argument("--num_questions", type=int, default=20, help="Number of questions that needs to be generated per context.")
    parser.add_argument(
        "--scope",
        type=int,
        choices=[1, -1],
        default=-1,
        help="Scope of the chapters to process. Allowed values: -1 (all next chapters), 1 (only the next chapter)."
    )
    args = parser.parse_args()

    chapters_path = args.chapters_path
    questions_path = args.questions_path
    num_questions = args.num_questions
    scope = args.scope

    # Configure Google Generative AI API
    GOOGLE_API_KEY = "AIzaSyCh4K2xZV9209xmJgjwxF-60xHdpaKU8vY"
    genai.configure(api_key=GOOGLE_API_KEY)

    # Load the model
    model = genai.GenerativeModel('models/gemini-2.0-flash-lite')

    # Get individual chapters
    with open(chapters_path) as f:
        chapters = json.load(f)
    T = len(chapters)
    print(f"Number of chapters: {T}")

    # Generate questions
    questions = []
    for t in range(T):
        history = chapters[t]

        # get prompt
        prompt = get_prompt(history, num_questions)

        print(f"Generating questions for chapter t={t} : {model.count_tokens(prompt)}")

        success = False
        while not success:
            # try:
            #     # Generate content with retry logic
            #     response = generate_text_with_retry(model, prompt)
            # except Exception as e:
            #     print(f"Error generating content. Retrying in 10 seconds...")
            #     time.sleep(10)

            response = generate_text_with_retry(model, prompt)

            # response = model.generate_content(prompt)
            q_list = parse_questions(response.text)
            if q_list is not None and len(q_list) == num_questions:
                if all([list(q.keys()) == KEYS for q in q_list]):
                    success = True
                    print(f"Successfully generated {num_questions} questions.")

        questions.append(
            {
                "chapter": t,
                "questions": q_list
            }
        )

        with open(questions_path, "w") as f:
            json.dump(questions, f, indent=4)