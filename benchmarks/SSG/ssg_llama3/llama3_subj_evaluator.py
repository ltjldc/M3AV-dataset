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

# Evaluate each generated answer and store results
evaluation_results = []
for item in data:
    generated_answer = item.get("generated_answer", "")
    if not generated_answer:
        continue
    
    # Define messages for evaluation with roles and structured content
#     evaluation_messages = [
#         {"role": "system", "content": "You are a helpful assistant evaluating academic presentations."},
#         {"role": "user", "content": f"""You will evaluate speech script A for one slide in an academic presentation. Rate it on a scale from 1 (worst) to 5 (best) in three aspects:

# 1. Fluency
#    - Objective: Assess sentence structure and grammar.
#    - Considerations: Are the sentences well-formed? Is the grammar correct and appropriate for an academic presentation?

# 2. Accuracy
#    - Objective: Check for factual correctness based on general knowledge.
#    - Considerations: Does Script A present accurate information consistent with widely accepted facts, established theories, findings, and methodologies in the relevant field? 

# 3. Coherence
#    - Objective: Evaluate the natural flow and readability of the script as a spoken presentation.
#    - Considerations: Does the script sound smooth and cohesive as a speech? Does it maintain a logical, engaging structure?

# Given the script below, provide scores and brief explanations:

# Script A:
# {generated_answer}

# Now, please start your evaluation:

# Fluency Score: [1-5]
# Reason: [Brief explanation]

# Accuracy Score: [1-5]
# Reason: [Brief explanation]

# Coherence Score: [1-5]
# Reason: [Brief explanation]"""}
#     ]
    evaluation_messages = [
        {"role": "system", "content": "You are a helpful assistant evaluating academic presentations."},
        {
        "role": "user",
        "content": f"""
        You will evaluate speech script A for one slide in an academic presentation. Rate it on a scale from 1 (worst) to 5 (best) in three aspects:

        1. **Focus and conciseness**
        - Does the script highlight key points effectively without unnecessary repetition?

        2. **Accuracy**
        - Is the information factually correct and aligned with established knowledge?

        3. **Clarity and Coherence**
        - Is the script logically structured and easy to follow?

        Given the script below, provide scores and brief explanations:

        **Script A:**
        {generated_answer}

        Now, please start your evaluation:

        - **Focus and conciseness Score:** [1-5]  
        *Reason:* [Brief explanation]

        - **Accuracy Score:** [1-5]  
        *Reason:* [Brief explanation]

        - **Clarity and Coherence Score:** [1-5]  
        *Reason:* [Brief explanation]
        """
        }
    ]

    # Generate evaluation using the model
    response = llm_pipeline(
        evaluation_messages,
        max_new_tokens=300,
        temperature=0.4,
        num_return_sequences=1,
        return_full_text=False
    )
    evaluation_text = response[0]['generated_text']
    print("Evaluation Text:", evaluation_text)
    
    # Store the evaluation result
    evaluation_results.append({
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "generated_answer": generated_answer,
        "evaluation": evaluation_text
    })

# Save the evaluation results to an output file
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_subj_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print(f"Processed data has been saved to {output_file}")
