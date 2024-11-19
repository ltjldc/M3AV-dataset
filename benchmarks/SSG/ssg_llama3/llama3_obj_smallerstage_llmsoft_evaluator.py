import json
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import re

# Set random seed for reproducibility
SEED_VALUE = 42
set_seed(SEED_VALUE)
# torch.manual_seed(SEED_VALUE)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED_VALUE)
# random.seed(SEED_VALUE)
# np.random.seed(SEED_VALUE)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Load LLaMA model and tokenizer
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/data/milsrg1/huggingface/cache/tl578/cache"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', cache_dir=CACHE_DIR)
model.eval()

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    pad_token_id=tokenizer.eos_token_id
)

# Load JSON data
input_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/sample,top_p=0.9,temperature=1.0_shortened.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Stage 1: Extract Information Units without Labels
extracted_units = []
for item in data:
    answer = item.get("answer", "")
    generated_answer = item.get("generated_answer", "")
    if not answer or not generated_answer:
        continue
    
    extraction_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""There are two scripts A and B. Please extract distinct information units critical to the academic content. Prioritize statements directly related to the research topic, such as definitions, key arguments, or distinctions, and ignore non-academic elements like greetings or transitions unless essential for understanding.

Script A (Answer):
{answer}

Script B (Generated Answer):
{generated_answer}

Provide the output in this format:

Script A Information Units:
- "Information Unit 1"
- "Information Unit 2"
...

Script B Information Units:
- "Information Unit 1"
- "Information Unit 2"
...

Do not include additional commentary or explanations before, within, or after the information units."""}
    ]

    response = llm_pipeline(
        extraction_messages,
        max_new_tokens=800,
        temperature=1.0,
        num_return_sequences=1,
        return_full_text=False
    )
    extraction_text = response[0]['generated_text']
    
    # Parse extracted text into units for Scripts A and B
    script_a_units, script_b_units = [], []
    current_section = None
    for line in extraction_text.splitlines():
        if line.startswith("Script A Information Units:"):
            current_section = "A"
        elif line.startswith("Script B Information Units:"):
            current_section = "B"
        elif line.startswith("-") and current_section == "A":
            script_a_units.append(line.strip("- ").strip())
        elif line.startswith("-") and current_section == "B":
            script_b_units.append(line.strip("- ").strip())
    
    extracted_units.append({
        "question": item.get("question", ""),
        "answer": answer,
        "generated_answer": generated_answer,
        "extraction": {
            "Script A Information Units": script_a_units,
            "Script B Information Units": script_b_units
        }
    })
    print(f"Extraction Result - Script A Units: {len(script_a_units)}, Script B Units: {len(script_b_units)}")

# Save intermediate results after Stage 1
intermediate_output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_info_units_llmsoft.json'
with open(intermediate_output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_units, f, indent=2, ensure_ascii=False)

print(f"Intermediate extraction results saved to {intermediate_output_file}")

# Stage 2: Coverage Scoring for Information Unit Matching
matching_results = []

for units in extracted_units:
    answer_units = units["extraction"]["Script A Information Units"]
    generated_units = units["extraction"]["Script B Information Units"]
    answer_context = units["answer"]
    generated_context = units["generated_answer"]

    # Step 2a: Calculate Coverage Score for each B Unit in Script A
    forward_matching_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""For each unit in Script B, assess how much of it is covered by Script A as a whole. Provide a score based on the extent of coverage as follows:

- 0: No overlap between the unit in Script B and the content in Script A.
- 0.2, 0.4, 0.6, 0.8: Partial coverage, with higher values indicating more significant overlap.
- 1: The unit in Script B is fully covered by the content in Script A.

Script A (Answer):
{answer_context}

Script B Information Units:
- {generated_units}

Output the results as follows:

Forward Coverage Score Analysis:
- Script B Unit: [Unit]
  - Coverage Score: [0 / 0.2 / 0.4 / 0.6 / 0.8 / 1]
"""}
    ]

    response = llm_pipeline(
        forward_matching_messages,
        max_new_tokens=800,
        temperature=0.3,
        num_return_sequences=1,
        return_full_text=False
    )
    forward_matching_text = response[0]['generated_text']

    # Step 2b: Calculate Coverage Score for each A Unit in Script B
    backward_matching_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""For each unit in Script A, assess how much of it is covered by Script B as a whole. Provide a score based on the extent of coverage as follows:

- 0: No overlap between the unit in Script A and the content in Script B.
- 0.2, 0.4, 0.6, 0.8: Partial coverage, with higher values indicating more significant overlap.
- 1: The unit in Script A is fully covered by the content in Script B.

Script B (Generated Answer):
{generated_context}

Script A Information Units:
- {answer_units}

Output the results as follows:

Backward Coverage Score Analysis:
- Script A Unit: [Unit]
  - Coverage Score: [0 / 0.2 / 0.4 / 0.6 / 0.8 / 1]
"""}
    ]

    response = llm_pipeline(
        backward_matching_messages,
        max_new_tokens=800,
        temperature=0.3,
        num_return_sequences=1,
        return_full_text=False
    )
    backward_matching_text = response[0]['generated_text']

    # Aggregate Results
    matching_summary = {
        "question": units["question"],
        "answer": units["answer"],
        "generated_answer": units["generated_answer"],
        "forward_coverage_score": forward_matching_text,
        "backward_coverage_score": backward_matching_text
    }
    matching_results.append(matching_summary)
    print(f"Coverage Score Summary:\n{forward_matching_text[:1000]}...\n{backward_matching_text[:1000]}...")

# Stage 3: Calculate Precision, Recall, and F1 Score using Coverage Scores
final_results = []
for result in matching_results:
    forward_coverage_text = result["forward_coverage_score"]
    backward_coverage_text = result["backward_coverage_score"]

    # Parse coverage scores for units in Script B (forward matching)
    total_b_units = 0
    b_coverage_sum = 0.0
    for line in forward_coverage_text.splitlines():
        match = re.search(r"Coverage Score:\s+(\d+(\.\d+)?)", line)
        if match:
            score = float(match.group(1))
            b_coverage_sum += score
            total_b_units += 1

    # Parse coverage scores for units in Script A (backward matching)
    total_a_units = 0
    a_coverage_sum = 0.0
    for line in backward_coverage_text.splitlines():
        match = re.search(r"Coverage Score:\s+(\d+(\.\d+)?)", line)
        if match:
            score = float(match.group(1))
            a_coverage_sum += score
            total_a_units += 1

    # Calculate Precision and Recall
    precision = b_coverage_sum / total_b_units if total_b_units > 0 else 0
    recall = a_coverage_sum / total_a_units if total_a_units > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    final_results.append({
        "question": result["question"],
        "answer": result["answer"],
        "generated_answer": result["generated_answer"],
        "forward_coverage_score": result["forward_coverage_score"],
        "backward_coverage_score": result["backward_coverage_score"],
        "metrics": {
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1 Score": round(f1_score, 2)
        }
    })
    print(f"Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

# Save the final results
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_obj_llmsoft_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"Final evaluation results saved to {output_file}")