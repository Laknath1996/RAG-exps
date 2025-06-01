import json
import re
import google.generativeai as genai
from google.api_core import retry, exceptions
import time

KEYS = ['question', 'A', 'B', 'C', 'D', 'answer', 'explanation']

def get_prompt(history, context, num_questions=10):
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
    return template.format(history=history, context=context, num_questions=num_questions)


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
        print(f"Encountered retryable error: {e}. Retrying...")
        return True
    return False

@retry.Retry(
    predicate=is_retryable_exception,
    # Initial delay before the first retry (seconds)
    initial=1.0,
    # Maximum delay between retries (seconds)
    maximum=60.0,
    # Multiplier for the delay (e.g., 2.0 means delay doubles each time)
    multiplier=2.0,
    # Total time to keep retrying before giving up (seconds)
    timeout=300.0,
    # Adding jitter to spread out retries
    # jitter=0.1 # You can add jitter if needed, the default might include some.
)

def generate_text_with_retry(model, prompt):
    response = model.generate_content(prompt)
    return response

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(description="Generate questions from a book.")
    parser.add_argument("--chapters_path", type=str, default="hp/hp1_2_chapters.json", help="Path to json file containing book chapters.")
    parser.add_argument("--questions_path", type=str, default="hp/hp1_2_questions.json", help="Path to json file that will contain the questions.")
    parser.add_argument("--num_questions", type=int, default=10, help="Number of questions that needs to be generated per context.")
    parser.add_argument(
        "--scope",
        type=int,
        choices=[1, -1],
        default=-1,
        help="Scope of the chapters to process. Allowed values: -1 (all next chapters), 1 (only the next chapter)."
    )
    parser.add_argument("--pause_iter", action="store_true", help="Pause after each iteration if set.")
    args = parser.parse_args()

    chapters_path = args.chapters_path
    questions_path = args.questions_path
    num_questions = args.num_questions
    scope = args.scope
    pause_iter = args.pause_iter

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
        k_end = T if scope == -1 else t+1+scope
        for k in range(t+1, k_end):
            context = chapters[k]

            # get prompt
            prompt = get_prompt(history, context, num_questions)

            print(f"t={t}, k={k}, total tokens : {model.count_tokens(prompt)}")

            success = False
            while not success:
                try:
                    # Generate content with retry logic
                    response = generate_text_with_retry(model, prompt)
                except Exception as e:
                    print(f"Error generating content: {e}")
                    time.sleep(10)

                # response = model.generate_content(prompt)
                q_list = parse_questions(response.text)
                if q_list is not None and len(q_list) == num_questions:
                    if all([list(q.keys()) == KEYS for q in q_list]):
                        success = True
                        print(f"Successfully generated {num_questions} questions.")

            questions.append(
                {
                    "history_upto": t,
                    "context": k,
                    "output": q_list
                }
            )

            with open(questions_path, "w") as f:
                json.dump(questions, f, indent=4)