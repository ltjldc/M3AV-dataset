import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from bert_score import score

SEED_VALUE = 42
set_seed(SEED_VALUE)
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

# Stage 1: Extract Information Units (same as before)
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
intermediate_output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_info_units_bert.json'
with open(intermediate_output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_units, f, indent=2, ensure_ascii=False)

# Stage 2: Information Unit Matching with BERTScore
matching_results = []
for units in extracted_units:
    answer_units = units["extraction"]["Script A Information Units"]
    generated_units = units["extraction"]["Script B Information Units"]

    # Initialize soft metrics
    tp = 0
    unique_b_score_sum = 0
    matched_a_units = {}

    # Calculate true positive and soft unique values for Script B
    for b_unit in generated_units:
        max_score, best_match = 0, None

        for a_unit in answer_units:
            _, _, f1 = score([b_unit], [a_unit], lang="en", verbose=False)
            f1_score = f1.mean().item()
            
            # Print BERTScore for comparison
            print(f"BERTScore for B Unit: '{b_unit}' vs A Unit: '{a_unit}' - Score: {f1_score:.4f}")

            if f1_score > max_score:
                max_score = f1_score
                best_match = a_unit

        tp += max_score
        unique_b_score_sum += (1 - max_score)
        matched_a_units[best_match] = max(matched_a_units.get(best_match, 0), max_score)

    # Calculate soft unique values for Script A
    unique_a_score_sum = sum(1 - score for score in matched_a_units.values())

    # Compute precision, recall, and F1 metrics
    fp = unique_b_score_sum
    fn = unique_a_score_sum

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Append results
    matching_results.append({
        "question": units["question"],
        "answer": units["answer"],
        "generated_answer": units["generated_answer"],
        "extraction": units["extraction"],
        "metrics": {
            "total_bert_tp": round(tp, 2),
            "unique_b_score_sum": round(unique_b_score_sum, 2),
            "unique_a_score_sum": round(unique_a_score_sum, 2),
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1 Score": round(f1_score, 2)
        }
    })
    print(f"Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}, "
          f"Total BERT TP: {tp:.2f}, Unique B Score Sum: {unique_b_score_sum:.2f}, Unique A Score Sum: {unique_a_score_sum:.2f}")

# Save the final results
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_bert_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(matching_results, f, indent=2, ensure_ascii=False)

print(f"Final evaluation results saved to {output_file}")
