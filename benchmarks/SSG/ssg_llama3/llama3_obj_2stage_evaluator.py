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
    
    # Define the messages for information extraction
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

    # Generate information units using the model
    response = llm_pipeline(
        extraction_messages,
        max_new_tokens=800,
        temperature=1.0,
        num_return_sequences=1,
        return_full_text=False
    )
    extraction_text = response[0]['generated_text']
    
    # Print the extraction result for verification
    print("Extraction Result:", extraction_text)
    
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
    
    # Store extracted units in structured format
    extracted_units.append({
        "question": item.get("question", ""),
        "answer": answer,
        "generated_answer": generated_answer,
        "extraction": {
            "Script A Information Units": script_a_units,
            "Script B Information Units": script_b_units
        }
    })
    
# Save the intermediate information unit extraction results to an output file
intermediate_output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_info_units.json'
with open(intermediate_output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_units, f, indent=2, ensure_ascii=False)

# Stage 2: Evaluate Precision and Recall
evaluation_results = []
for units in extracted_units:
    # Retrieve previously extracted units for evaluation
    answer_units = units["extraction"].get("Script A Information Units", [])
    generated_units = units["extraction"].get("Script B Information Units", [])

    # Format the information units for insertion into the evaluation prompt
    formatted_answer_units = "\n".join([f"- {unit}" for unit in answer_units])
    formatted_generated_units = "\n".join([f"- {unit}" for unit in generated_units])

    # Define the messages for precision and recall evaluation
    evaluation_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""Evaluate the precision, recall and F1 score of information units in Script B relative to Script A.

Script A Information Units:
{formatted_answer_units}

Script B Information Units:
{formatted_generated_units}

1. Analyze Information Units:
   - Exact Match: Units conveying the same content with near-identical meaning.
   - Partial Match: Similar ideas with minor phrasing differences or added context (count each as 0.5).
   - Unique Units: Units found only in one script with no equivalent in the other.
   
2. Calculate Metrics:
   - True Positives (TP): Exact and partial matches.
   - False Positives (FP): Unique Units only in Script B.
   - False Negatives (FN): Unique Units only in Script A.
   
   Formulas:
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

3. Output in this format:
Recall Evaluation:
- Relevant information units missing in Script B: [Count/Total in Script A]

Precision Evaluation:
- Irrelevant information units in Script B: [Count/Total in Script B]

F1 score: [Value]"""}
    ]

    # Generate evaluation using the model
    response = llm_pipeline(
        evaluation_messages,
        max_new_tokens=800,
        temperature=0.3,
        num_return_sequences=1,
        return_full_text=False
    )
    evaluation_text = response[0]['generated_text']
    
    # Print the evaluation for verification
    print("Evaluation Text:", evaluation_text)
    
    # Store evaluation result
    evaluation_results.append({
        "question": units["question"],
        "answer": units["answer"],
        "generated_answer": units["generated_answer"],
        "extraction": units["extraction"],
        "evaluation": evaluation_text
    })

# Save the evaluation results to an output file
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_obj_2stage_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
