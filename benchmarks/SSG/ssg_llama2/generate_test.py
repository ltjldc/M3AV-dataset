import torch
from torch.cuda.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
from tqdm import tqdm

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

@torch.no_grad()
def generate(prompt_tokens, model, tokenizer, args):
    set_seed(args['seed'])
    tokens = prompt_tokens
    past_key_values = None
    for _ in tqdm(range(args['max_gen_len'])):
        with autocast(dtype=torch.bfloat16):
            if past_key_values is not None:
                output = model(input_ids=tokens[:, -1:], use_cache=True, past_key_values=past_key_values)
            else:
                output = model(input_ids=tokens, use_cache=True)
        
        past_key_values = output.past_key_values
        logits = output.logits[:, -1]
        
        if args['method'] == 'greedy_search':
            next_token = torch.argmax(logits, dim=-1)
        elif args['method'] == 'sample':
            probs = torch.softmax(logits / args['temperature'], dim=-1)
            next_token = sample_top_p(probs, args['top_p'])
        else:
            raise NotImplementedError
        
        next_token = next_token.reshape(1, -1)
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if next_token[0] == tokenizer.eos_token_id:
            break
    
    tokens = tokens.tolist()
    for i in range(len(tokens)):
        tokens[i] = tokens[i][len(prompt_tokens[i]):]
    
    decoded = []
    for t in tokens:
        try:
            t = t[: t.index(tokenizer.eos_token_id)]
        except ValueError:
            pass
        decoded.append(tokenizer.decode(t))
    return decoded

def main(input_text, model_name, cache_dir, exp_dir, max_gen_len, method='sample', top_p=0.9, temperature=1.0):
    # Configuration
    args = {
        'max_gen_len': max_gen_len,
        'method': method,
        'top_p': top_p,
        'temperature': temperature,
        'seed': 42
    }
    set_seed(args['seed'])
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', cache_dir=cache_dir)
    lora_model = PeftModel.from_pretrained(base_model, exp_dir, torch_dtype=torch.bfloat16)
    model = lora_model.merge_and_unload()
    model.eval().cuda()

    # Tokenize input
    question_tokens = tokenizer.encode(input_text, return_tensors='pt').cuda()
    
    # Add bos_token_id at the beginning
    bos_token = torch.tensor([[tokenizer.bos_token_id]]).cuda()
    # prompt_tokens = torch.cat((bos_token, question_tokens), dim=1)
    prompt_tokens = question_tokens
    print(tokenizer.decode(prompt_tokens[0]))
    decoded = generate(prompt_tokens, model, tokenizer, args)
    
    return decoded[0]

# Example usage
input_text = "{user_message} [/INST] # There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text in the given single slide.\n# Slide text:\nThe\nSemi-structured interviews with 17 participants\nTable 1. Participants\nGender\nDevice Watch GT Honor Watch Magic Honor Watch Magic, Watch GT Huawei B5 bracelet Honor Watch Magic Honor Watch Magic Watch GT Honor Watch Magic Watch GT Honor Watch Magic Watch GT Watch GT, Garmin vivosmart4 Watch GT Honor Watch Magic Honor Watch Magic, Watch GT Watch GT Garmin Forerunner 645 Music\nOccupation Exhibition Salesman Government Worker IT Manufacturing Trainee Technical Developer Design and Operation Worker Programmer Wearable Health Worker Business Operator Government Worker Programmer Unemployed Exercise physiology worker Customer service Unemployed Programmer Advertising designer Student\nLocation Shanghai Jilin Guangdong Beijing Henan Beijing Shenzhen Henan Shanghai Beijing Shenyang Finland and China Shanghai Shanghai Xian Shanghai Guangzhou\nID\nAge\nWe had been following up with some participants after the interviews.\n# Speech text:\n"

model_name = "meta-llama/Llama-2-7b-hf"
cache_dir = "/data/milsrg1/huggingface/cache/tl578/cache"
exp_dir = "ssg_llama2/alog/l2p/best_model"  # Example directory for LoRA model
max_gen_len = 300

output = main(input_text, model_name, cache_dir, exp_dir, max_gen_len)
print(output)
