import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load LLaMA model and tokenizer
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/data/milsrg1/huggingface/cache/tl578/cache"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', cache_dir=CACHE_DIR)
device = torch.device('cuda')
model.eval()

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    pad_token_id=tokenizer.eos_token_id
)

# Load the JSON file with answer and generated answers
input_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/sample,top_p=0.9,temperature=1.0_shortened.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Evaluate each pair and store results
evaluation_results = []
for item in data:
    answer = item.get("answer", "")
    generated_answer = item.get("generated_answer", "")
    if not answer or not generated_answer:
        continue

    # Define the messages for evaluation
    evaluation_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""You will evaluate two versions of a speech script (Script A and Script B) for one slide in an academic presentation. Script A is the reference answer, and Script B is the generated answer. Your goal is to identify information units in both scripts and compare them for relevance and consistency.

1. Segment Script A and Script B into distinct information units. Each information unit should represent a single factual point or distinct idea.

2. Count the number of information units in Script A that are missing in Script B (Recall).
3. Count the number of information units in Script B that are not relevant to Script A (Precision).

Script A (Answer):
{answer}

Script B (Generated Answer):
{generated_answer}

Provide your output in this format:

Script A (Answer) Information Units:
[List information units here]

Script B (Generated Answer) Information Units:
[List information units here]

Recall Evaluation:
- Number of relevant information units missing in Script B: [Count]

Precision Evaluation:
- Number of irrelevant information units in Script B: [Count]

Please output your evaluation clearly without any additional commentary."""}
    ]

    # Generate evaluation using the model
    response = llm_pipeline(
        evaluation_messages,
        max_new_tokens=1000,
        temperature=0.3,
        num_return_sequences=1,
        return_full_text=False
    )
    evaluation_text = response[0]['generated_text']
    
    # Print the evaluation for verification
    print("Evaluation Text:", evaluation_text)
    
    # Store the evaluation result
    evaluation_results.append({
        "question": item.get("question", ""),
        "answer": answer,
        "generated_answer": generated_answer,
        "evaluation": evaluation_text
    })

# Save the evaluation results to an output file
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_obj_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
