import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the input and output file paths
input_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/sample,top_p=0.9,temperature=1.0.json'    # Replace with your input file name
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/sample,top_p=0.9,temperature=1.0_shortened.json'   # Replace with your desired output file name

# Define the model and tokenizer
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # Ensure this is the correct Llama3 model name
CACHE_DIR = "/data/milsrg1/huggingface/cache/tl578/cache"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', cache_dir=CACHE_DIR)

# Set the pad_token to eos_token if it is not set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Set the device
model.eval()
model.cuda()
device = torch.device('cuda')

# Function to process the text using Llama
def process_text_with_llama(input_text):
    prompt = f"""You are an assistant that helps to clean scripts.
The user provides a text that may contain headers or extra text before the actual script.
Please extract and return only the main script, removing any headers or front matter.

Here is the text:

{input_text}

The cleaned script is:
"""
    # Generate inputs with padding and attention mask
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    
    # Generate output with attention mask and pad_token_id set
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=inputs['input_ids'].shape[1] + 500,
        do_sample=True,
        temperature=0.7,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated response after "The cleaned script is:"
    cleaned_text = generated_text.split("The cleaned script is:")[-1].strip()
    return cleaned_text

# Load the JSON data from the input file
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process each item in the list
for item in data:
    # Get the generated_answer list
    generated_answers = item.get('generated_answer', [])
    cleaned_answers = []
    

    # Process and print each cleaned answer
    cleaned_text = process_text_with_llama(generated_answers)
    
    cleaned_answers.append(cleaned_text)
    
    # Update the generated_answer field with the cleaned text
    item['generated_answer'] = cleaned_answers[0]
    print("Cleaned Answer:", cleaned_answers[0])  # 实时打印出每个生成的 cleaned_text

# Save the modified data to the output file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Processed data has been saved to {output_file}")