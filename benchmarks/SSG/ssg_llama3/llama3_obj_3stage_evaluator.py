import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

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
        {"role": "user", "content": f"""There are two scripts A and B. Please extract distinct statements or concepts critical to the academic content. For extraction, prioritize statements directly related to the research topic, such as definitions, key arguments, or distinctions; disregard non-academic elements like greetings, transitions, or personal expressions unless they are critical to understanding the academic material.

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
    
    # Parse the extracted text to split into Script A and Script B units
    script_a_units = []
    script_b_units = []
    current_section = None
    for line in extraction_text.splitlines():
        if line.strip().startswith("Script A Information Units:"):
            current_section = "A"
        elif line.strip().startswith("Script B Information Units:"):
            current_section = "B"
        elif line.strip().startswith("-") and current_section == "A":
            script_a_units.append(line.strip().lstrip("- ").strip())
        elif line.strip().startswith("-") and current_section == "B":
            script_b_units.append(line.strip().lstrip("- ").strip())
    
    extracted_units.append({
        "question": item.get("question", ""),
        "answer": answer,
        "generated_answer": generated_answer,
        "extraction": {
            "Script A Information Units": script_a_units,
            "Script B Information Units": script_b_units
        }
    })

    # Print concise intermediate extraction result
    print(f"Question: {item.get('question', '')[:50]}...")  # Show truncated question for readability
    print(f"Script A Units (count): {len(script_a_units)}")
    print(f"Script B Units (count): {len(script_b_units)}")
    print("----------")

# Save intermediate results
intermediate_output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_info_units.json'
with open(intermediate_output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_units, f, indent=2, ensure_ascii=False)

# Stage 2: Information Unit Matching
matching_results = []
for units in extracted_units:
    answer_units = units["extraction"].get("Script A Information Units", [])
    generated_units = units["extraction"].get("Script B Information Units", [])
    
    formatted_answer_units = "\n".join([f"- {unit}" for unit in answer_units])
    formatted_generated_units = "\n".join([f"- {unit}" for unit in generated_units])

    matching_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""Compare each information unit in Script B against Script A. For each unit in Script B, indicate if there is an **Exact Match**, **Partial Match**, or **No Match** in Script A.

Script A Information Units:
{formatted_answer_units}

Script B Information Units:
{formatted_generated_units}

Output in this format:
- Script B Unit: [Unit]
  - Match Status: [Exact Match / Partial Match / No Match]
  - Corresponding Script A Unit (if any): [Script A Unit or "None"]
"""}
    ]

    response = llm_pipeline(
        matching_messages,
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

    # Print concise matching result
    print(f"Matching for Question: {units.get('question', '')[:50]}...")
    print(f"Matching Summary:\n{matching_text[:800]}...")  # Show first 200 chars for readability
    print("----------")

# Stage 3: Calculate Precision, Recall, and F1 Score
final_results = []
for result in matching_results:
    matching_text = result["matching"]

    # Count the matches based on the matching_text analysis
    exact_matches = matching_text.count("Exact Match")
    partial_matches = matching_text.count("Partial Match") * 0.5
    unique_b = matching_text.count("No Match")
    unique_a = len(result["extraction"]["Script A Information Units"]) - (exact_matches + partial_matches)

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
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1 Score": round(f1_score, 2)
        }
    })

    # Print concise metric results
    print(f"Metrics for Question: {result.get('question', '')[:50]}...")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}")
    print("----------")

# Save the final results to an output file
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_final_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"Final evaluation results saved to {output_file}")
