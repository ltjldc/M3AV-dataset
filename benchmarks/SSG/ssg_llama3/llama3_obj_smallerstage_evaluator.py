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

# Stage 1: Extract Information Units
extracted_units = []
for item in data:
    answer = item.get("answer", "")
    generated_answer = item.get("generated_answer", "")
    if not answer or not generated_answer:
        continue
    
    extraction_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""There are two scripts A and B. Please extract distinct information units critical to the academic content. For extraction, prioritize statements directly related to the research topic, such as definitions, key arguments, or distinctions; disregard non-academic elements like greetings, transitions, or personal expressions unless they are critical to understanding the academic material.

        Script A (Answer):
        {answer}

        Script B (Generated Answer):
        {generated_answer}

        Provide the output strictly in this format for both scripts:

        Script A Information Units:
        - [Label 1]: "Information Unit 1"
        - [Label 2]: "Information Unit 2"
        ...

        Script B Information Units:
        - [Label 1]: "Information Unit 1"
        - [Label 2]: "Information Unit 2"
        ...

        For each label, briefly identify the main theme or topic of the information unit (such as a key argument or concept). Do not include any additional commentary, explanations before, within, or after the extracted information units."""}

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

# Save intermediate results
intermediate_output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_info_units_smallerstage.json'
with open(intermediate_output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_units, f, indent=2, ensure_ascii=False)

# Stage 2: Information Unit Matching with Bidirectional Context Check (in smaller steps)
matching_results = []

for units in extracted_units:
    answer_units = units["extraction"]["Script A Information Units"]
    generated_units = units["extraction"]["Script B Information Units"]
    answer_context = units["answer"]
    generated_context = units["generated_answer"]

    # Step 2a: Forward Matching (Script B to Script A)
    forward_matching_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""Check each unit in Script B against all units in Script A and categorize them as follows:

       - Exact Match: The unit in Script B has highly similar content to a unit in Script A, with only minor phrasing differences but no additional or omitted meaning.
       - Partial Match: The unit in Script B shares some similar meaning with a unit in Script A but also includes additional information or omits some parts of the meaning.
       - No Match: No unit in Script A matches the content of the Script B unit.

       Context for Script A:
       {answer_context}

       Script A Information Units:
       - {answer_units}

       Script B Information Units:
       - {generated_units}

       Output the results as follows:

       Forward Match Analysis:
       - Script B Unit: [Unit]
         - Match Status in Script A: [Exact Match / Partial Match / No Match]
         - Corresponding Script A Unit (if any): [Script A Unit or "None"]
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

    # Step 2b: Backward Matching (Script A to Script B)
    backward_matching_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""Check each unit in Script A against all units in Script B and categorize them as follows:

       - Exact Match: The unit in Script A has highly similar content to a unit in Script B, with only minor phrasing differences but no additional or omitted meaning.
       - Partial Match: The unit in Script A shares some similar meaning with a unit in Script B but also includes additional information or omits some parts of the meaning.
       - No Match: No unit in Script B matches the content of the Script A unit.

       Context for Script B:
       {generated_context}

       Script B Information Units:
       - {generated_units}

       Script A Information Units:
       - {answer_units}

       Output the results as follows:

       Backward Match Analysis:
       - Script A Unit: [Unit]
         - Match Status in Script B: [Exact Match / Partial Match / No Match]
         - Corresponding Script B Unit (if any): [Script B Unit or "None"]
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

    # Step 2c: Aggregate Results
    matching_summary = {
        "question": units["question"],
        "answer": units["answer"],
        "generated_answer": units["generated_answer"],
        "forward_matching": forward_matching_text,
        "backward_matching": backward_matching_text
    }
    matching_results.append(matching_summary)
    print(f"Matching Summary:\n{forward_matching_text[:400]}...\n{backward_matching_text[:400]}...")

# Stage 3: Calculate Precision, Recall, and F1 Score
final_results = []
for result in matching_results:
    forward_matching_text = result["forward_matching"]
    backward_matching_text = result["backward_matching"]

    # Initialize counters
    exact_matches = 0
    partial_matches = 0
    unique_b = 0
    unique_a = 0

    # Parse forward matching results (only count Script B -> Script A matches)
    for line in forward_matching_text.splitlines():
        if "Exact Match" in line:
            exact_matches += 1
        elif "Partial Match" in line:
            partial_matches += 0.5  # Count partial matches as 0.5
        elif "No Match" in line:
            unique_b += 1  # Unique units in Script B

    # Parse backward matching results (only count unique units in Script A)
    for line in backward_matching_text.splitlines():
        if "No Match" in line:
            unique_a += 1  # Unique units in Script A

    # Compute metrics
    tp = exact_matches + partial_matches
    fp = unique_b
    fn = unique_a

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    final_results.append({
        "question": result["question"],
        "answer": result["answer"],
        "generated_answer": result["generated_answer"],
        "forward_matching": result["forward_matching"],
        "backward_matching": result["backward_matching"],
        "metrics": {
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "unique_b": unique_b,
            "unique_a": unique_a,
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1 Score": round(f1_score, 2)
        }
    })
    print(f"Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")

# Save the final results
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_obj_smallerstage_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"Final evaluation results saved to {output_file}")
