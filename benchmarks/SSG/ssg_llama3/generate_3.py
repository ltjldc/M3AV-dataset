import copy
from glob import glob
import json
import os, random, time
import subprocess
from os.path import join as pjoin
from tqdm import tqdm
from pprint import pprint

import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer

from args import get_args
from dataset import get_dataset

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
def generate(
    prompt_tokens,
    args
):
    set_seed(args.seed)
    tokens = prompt_tokens
    past_key_values = None
    for _ in tqdm(range(args.max_gen_len)):
        with autocast(dtype=torch.bfloat16):
            if past_key_values is not None:
                output = model(
                    input_ids=tokens[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values
                )
            else:
                output = model(
                    input_ids=tokens,
                    use_cache=True,
                    past_key_values=past_key_values
                )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1]
        
        if args.method == 'greedy_search':
            next_token = torch.argmax(logits, dim=-1)
        elif args.method == 'sample':
            probs = torch.softmax(logits / args.temperature, dim=-1)
            next_token = sample_top_p(probs, args.top_p)
        else:
            raise NotImplementedError
        next_token = next_token.reshape(1, -1)
        tokens = torch.cat([tokens, next_token], dim=1)
        
        if next_token[0] == tokenizer.eos_token_id:
            break
    tokens = tokens.tolist()

    for i in range(len(tokens)):
        tokens[i] = tokens[i][len(prompt_tokens[i]): ]
        
    decoded = []
    for i, t in enumerate(tokens):
        # cut to eos token if any
        try:
            t = t[: t.index(tokenizer.eos_token_id)]
        except ValueError:
            pass
        decoded.append(tokenizer.decode(t))

    return decoded

if __name__ == "__main__":
    _, args = get_args('generate')
    assert os.path.exists(pjoin('ssg_llama2/alog', args.exp, '.is_done'))
    
    with open(pjoin('ssg_llama2/alog', args.exp, 'common_args.json')) as f:
        common_args = json.load(f)
    args.__dict__.update(common_args)

    print('=' * 89)
    pprint(args.__dict__)
    set_seed(args.seed)

    # 定义模型名称和缓存目录
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    CACHE_DIR = "/data/milsrg1/huggingface/cache/tl578/cache"

    # 使用 Hugging Face 的 AutoTokenizer 和 AutoModelForCausalLM 加载模型和分词器，并指定 cache_dir
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
    model.eval()
    model.cuda()

    gen_data_text = get_dataset(data_split='test', args=args, raw_text=True)
    gen_data_token = get_dataset(data_split='test', args=args, raw_text=False)
    all_index = list(range(len(gen_data_token)))

    method_str = f'{args.method}'
    if args.method == 'sample':
        method_str += f',top_p={args.top_p},temperature={args.temperature}'

    for index in tqdm(all_index):
        question_tokens = gen_data_token[index]['question']
        question_tokens.insert(0, tokenizer.bos_token_id)
        prompt_tokens = torch.as_tensor(question_tokens).unsqueeze(0).long().cuda()
        decoded = generate(prompt_tokens, args)
        gen_data_text[index]['generated_answer'] = decoded[0]
        print(decoded[0])

    os.makedirs(pjoin('ssg_llama2/alog', args.exp, 'gen'), exist_ok=True)
    with open(pjoin('ssg_llama2/alog', args.exp, 'gen', f'{method_str}.json'), 'w', encoding='utf-8') as f:
        json.dump(gen_data_text, f, indent=2, ensure_ascii=False)
