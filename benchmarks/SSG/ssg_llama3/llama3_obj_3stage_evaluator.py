import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import re

set_seed(42)

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
- [Information Unit 1]
- [Information Unit 2]
...

Script B Information Units:
- [Information Unit 1]
- [Information Unit 2]
...

Do not include any additional commentary, explanations before, within, or after the extracted information units."""}
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
intermediate_output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_info_units_3stage.json'
with open(intermediate_output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_units, f, indent=2, ensure_ascii=False)

# Stage 2: Information Unit Matching with Bidirectional Context Check
matching_results = []
for units in extracted_units:
    answer_units = units["extraction"]["Script A Information Units"]
    generated_units = units["extraction"]["Script B Information Units"]
    answer_context = units["answer"]
    generated_context = units["generated_answer"]

    # Define bidirectional matching messages
    bidirectional_matching_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""Analyze information units in Scripts A and B. For each unit in Script B, indicate if it has an **Exact Match**, **Partial Match**, or **No Match** in Script A. Similarly, for each unit in Script A, do the same in relation to Script B.

Criteria for Match Status:
   - Exact Match: Units conveying the highly similar or identical contents, allowing for minor phrasing differences but without significant additions or omissions in meaning.
   - Partial Match: Units share similar ideas with noticeable content differences, such as partial additions, omissions or rephrasing that affect the conveyed meaning (count each as 0.5).
   - Unique Units (No Match): Units found only in one script with no equivalent in the other.

Context for Script A:
{answer_context}

Context for Script B:
{generated_context}

Script A Information Units:
- {answer_units}

Script B Information Units:
- {generated_units}

Output in this format:
Match Analysis:
- Script B Unit: [Unit]
  - Match Status in Script A: [Exact Match / Partial Match / No Match]
  - Corresponding Script A Unit (if any): [Script A Unit or "None"]
- Script A Unit: [Unit]
  - Match Status in Script B: [Exact Match / Partial Match / No Match]
  - Corresponding Script B Unit (if any): [Script B Unit or "None"]
"""}
    ]

    response = llm_pipeline(
        bidirectional_matching_messages,
        max_new_tokens=800,
        temperature=0.3,
        num_return_sequences=1,
        return_full_text=False
    )
    matching_text = response[0]['generated_text']
    matching_results.append({
        "question": units["question"],
        "answer": units["answer"],
        "generated_answer": units["generated_answer"],
        "extraction": units["extraction"],
        "matching": matching_text
    })
    print(f"Matching Summary:\n{matching_text[:800]}...")  # Display part of matching output

# Stage 3: Calculate Precision, Recall, and F1 Score
final_results = []
for result in matching_results:
    matching_text = result["matching"]

    # Initialize counters
    exact_matches = 0
    partial_matches = 0
    unique_b = 0
    unique_a = 0

    # Patterns for parsing matches in the matching text
    a_unit_pattern = r"- Script A Unit: (.+)"
    b_unit_pattern = r"- Script B Unit: (.+)"
    match_status_pattern = r"Match Status in (Script A|Script B): (Exact Match|Partial Match|No Match)"
    corresponding_unit_pattern = r"Corresponding (Script A|Script B) Unit: (.+)"

    # Parse blocks of the matching text
    matching_blocks = matching_text.split("\n\n")
    for block in matching_blocks:
        script_unit = None
        match_status = None
        corresponding_unit = None
        context = None

        # Extract each line in a block and process before overwriting
        for line in block.splitlines():
            if re.search(a_unit_pattern, line):
                script_unit = re.search(a_unit_pattern, line).group(1).strip()
                context = "A"
            elif re.search(b_unit_pattern, line):
                script_unit = re.search(b_unit_pattern, line).group(1).strip()
                context = "B"
            elif re.search(match_status_pattern, line):
                match_status = re.search(match_status_pattern, line).group(2).strip()

                # Process matches immediately after updating match_status
                if context == "B":
                    if match_status == "Exact Match":
                        exact_matches += 1
                    elif match_status == "Partial Match":
                        partial_matches += 0.5
                        unique_b += 0.5  # Count `Partial Match` as 0.5 for `Script B` units
                    elif match_status == "No Match":
                        unique_b += 1
                elif context == "A" and match_status == "No Match":
                    unique_a += 1  # Only count `No Match` for `Script A` units as unique_a
                elif context == "A" and match_status == "Partial Match":
                    unique_a += 0.5
            elif re.search(corresponding_unit_pattern, line):
                corresponding_unit = re.search(corresponding_unit_pattern, line).group(2).strip()

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
        "extraction": result["extraction"],
        "matching": result["matching"],
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
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_obj_3stage_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"Final evaluation results saved to {output_file}")