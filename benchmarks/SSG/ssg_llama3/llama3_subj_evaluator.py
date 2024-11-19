import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the model and tokenizer from cache
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/data/milsrg1/huggingface/cache/tl578/cache"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', cache_dir=CACHE_DIR)
model.eval()

# Set up the pipeline for generation
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    pad_token_id=tokenizer.eos_token_id
)

# Load the JSON file with generated answers
input_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/sample,top_p=0.9,temperature=1.0.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Define evaluation prompts for each criterion with clear output instructions and a scale of 1-10
criteria_prompts = {
    "focus": """
        Focus (1-10):  
        Definition: Evaluates whether the script remains centered on its main idea or topic without introducing irrelevant content.  
        Sub-Criteria:  
        - Relevance of Content: Are all points, examples, and details directly tied to the central idea?  
        - Clarity of Objective: Is the main purpose or argument of the script clearly defined and maintained throughout?  
        - Avoidance of Tangents: Are there minimal deviations or distractions from the main topic?  

        **Instruction:** Strictly follow this format:  
        - First line: "Score: [1-10]"  
        Any deviation from this format will be considered incorrect.
    """,
    "coherence": """
        Coherence (1-10):  
        Definition: Assesses the logical organization and connectivity of ideas, ensuring a smooth and unified narrative.  
        Sub-Criteria:  
        - Logical Progression: Are ideas presented in a sequence that makes sense and builds logically from one point to the next?  
        - Internal Consistency: Are there no contradictions, overlaps, or repeated points that disrupt the overall logic?  
        - Quality of Transitions: Do transitions between sentences and sections enhance the flow, avoiding abrupt changes?  

        **Instruction:** Strictly follow this format:  
        - First line: "Score: [1-10]"  
        Any deviation from this format will be considered incorrect.
    """,
    "fluency": """
        Fluency (1-10):  
        Definition: Evaluates the linguistic quality and readability of the script, focusing on how effectively it communicates.  
        Sub-Criteria:  
        - Language Accuracy: Are sentences grammatically correct and free of awkward phrasing or errors?  
        - Clarity of Expression: Are ideas expressed clearly, with appropriate vocabulary and sentence structures?  
        - Pacing and Rhythm: Is the length and structure of sentences and paragraphs balanced, making the script easy to follow and deliver?  

        **Instruction:** Strictly follow this format:  
        - First line: "Score: [1-10]"  
        Any deviation from this format will be considered incorrect.
    """
}

# Evaluate each generated answer and store results
evaluation_results = []
for item in data:
    generated_answer = item.get("generated_answer", "")
    if not generated_answer:
        continue

    criterion_scores = {}
    for criterion, prompt in criteria_prompts.items():
        evaluation_message = [
            {"role": "system", "content": "You are a helpful assistant evaluating academic presentations."},
            {
                "role": "user",
                "content": f"""
                Script A:  
                {generated_answer}  

                {prompt}
                """
            }
        ]
        
        # Generate evaluation using the model
        response = llm_pipeline(
            evaluation_message,
            max_new_tokens=150,
            temperature=0.8,
            num_return_sequences=1,
            return_full_text=False
        )
        
        evaluation_text = response[0]['generated_text']
        print(f"{criterion} Evaluation Text:", evaluation_text)

        # Validate and parse the output to extract the score
        score = None
        lines = evaluation_text.splitlines()
        if len(lines) > 0 and lines[0].startswith("Score:"):
            try:
                score = int(lines[0].split("Score:")[-1].strip())
            except ValueError:
                pass  # Log invalid score format if needed

        criterion_scores[criterion] = {
            "full_output": evaluation_text,
            "score": score
        }

    # Store the evaluation result
    evaluation_results.append({
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "generated_answer": generated_answer,
        "evaluation": criterion_scores
    })

# Save the evaluation results to an output file
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_subj_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print(f"Processed data has been saved to {output_file}")
