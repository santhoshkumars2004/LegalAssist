import openai

# Set your OpenAI GPT-3 API key here
openai.api_key = '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'

def generate_answer(prompt):
    model_prompt = f"Given the prompt:\n\n{prompt}\n\nProvide a solution:\n"
    
    response = openai.Completion.create(
        engine="gpt-4.0-turbo",  # Use the GPT-4 engine
        prompt=model_prompt,
        max_tokens=500,  # Adjust as needed
        temperature=0.7,  # Adjust for creativity vs. precision
        stop=None  # Add custom stop conditions if needed
    )

    solution = response.choices[0].text.strip()
    return solution

def process_file(gen_inp):
    with open(gen_inp, 'r') as file:
        prompts = file.readlines()

    solutions = []
    for prompt in prompts:
        solution = generate_answer(prompt)
        solutions.append(solution)

    return solutions

