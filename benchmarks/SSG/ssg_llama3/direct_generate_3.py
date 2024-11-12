import json
import os
import torch
from os.path import join as pjoin
from tqdm import tqdm
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from args import get_args
from dataset import get_dataset
from torch.cuda.amp import autocast
from transformers import pipeline

def test_with_pipeline(model, tokenizer, device):
    # 创建 Hugging Face 的生成 pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16,
        pad_token_id=tokenizer.eos_token_id
    )

    # 定义测试提示
    test_prompt = (
        "You are a professional speaker. Based on the provided slide text and related sentences from the paper, "
        "generate the corresponding speech text.\n\n"
        "Slide Text:\n"
        "Data Engagement Reconsidered: A Study of Automatic Stress Tracking Technology in Use\n"
        "Xianghua Ding 1, Shuhan Wei 1, Xinning Gui 2, Ning Gu 1, Peng Zhang\n"
        "[1] Cooperative and Information Systems Lab, School of Computer Science, Fudan University, Shanghai, China\n"
        "[2] College of Information Sciences and Technology, Pennsylvania State University, PA, USA\n"
        "CHI202I\n"
        "PennState\n\n"
        "Related Sentences in the Paper:\n"
        "Below, we turn our focus to the challenges of stress-tracking data engagement we uncovered from the study.\n"
        "Here we review the self-tracking work that is related to data engagement.\n"
        "As shown in our study, one challenge of stress-tracking data engagement comes from how the data is encountered.\n\n"
        "Speech Text:\n"
    )

    # 生成文本
    generated = generator(test_prompt, max_length=256, temperature=0.7, top_p=0.8, num_return_sequences=1)
    print(f"Pipeline Generated Answer: {generated[0]['generated_text']}")

# 定义 top-p 采样函数
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)  # 形状：[batch_size, 1]
    next_token = torch.gather(probs_idx, -1, next_token)      # 形状：[batch_size, 1]
    return next_token

# 定义生成函数
# @torch.no_grad()
# def generate(prompt_tokens, model, tokenizer, args):
#     """生成文本的函数"""
#     tokens = prompt_tokens  # 形状：[batch_size, seq_len]
#     eos_token_id = tokenizer.eos_token_id
#     past_key_values = None
#     generated_tokens = []
    
#     for _ in range(args.max_gen_len):
#         # 使用更新后的 autocast 方式
#         with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
#             output = model(
#                 input_ids=tokens[:, -1:],  # 只输入最后一个 token
#                 use_cache=True,
#                 past_key_values=past_key_values
#             )
#         logits = output.logits[:, -1, :]  # 形状：[batch_size, vocab_size]
#         past_key_values = output.past_key_values

#         if args.method == 'greedy_search':
#             # 获取概率最高的 token
#             next_token = torch.argmax(logits, dim=-1, keepdim=True)  # 形状：[batch_size, 1]
#         elif args.method == 'sample':
#             probs = torch.softmax(logits / args.temperature, dim=-1)
#             next_token = sample_top_p(probs, args.top_p)  # 形状：[batch_size, 1]
#         else:
#             raise NotImplementedError(f"Method {args.method} is not implemented.")

#         # 将新生成的 token 拼接到序列中
#         tokens = torch.cat([tokens, next_token], dim=1)  # 形状：[batch_size, seq_len + 1]
#         generated_tokens.append(next_token.item())

#         # 检查是否生成了 EOS token
#         if eos_token_id is not None and (next_token == eos_token_id).all():
#             break

#     # 获取生成的部分
#     generated_tokens_tensor = tokens[:, prompt_tokens.size(1):]  # 形状：[batch_size, generated_seq_len]
#     generated_text = tokenizer.decode(generated_tokens_tensor[0], skip_special_tokens=True)

#     # 调试信息：打印生成的 token ID 和生成的文本
#     print(f"Generated Tokens IDs: {generated_tokens}")
#     print(f"Generated Text: {generated_text}")

#     return generated_text
@torch.no_grad()
def generate(
    prompt_tokens,
    model,
    tokenizer,
    args
) :
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
        logits = output.logits[: ,-1 ]
        
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
        # cut to eos tok if any
        try:
            t = t[: t.index(tokenizer.eos_token_id)]
        except ValueError:
            pass
        decoded.append(tokenizer.decode(t))

    return decoded

def main():
    # 主执行块
    common_args, args = get_args('generate')
    args.output_dir = pjoin('ssg_llama2/alog', args.exp)
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置随机种子，仅调用一次
    set_seed(args.seed)

    # 定义模型和分词器
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"  # 确保这是正确的 Llama3 模型名称
    CACHE_DIR = "/data/milsrg1/huggingface/cache/tl578/cache"

    # 使用 AutoTokenizer 加载分词器
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, padding_side="left")

    # 设置 pad_token 为 eos_token（大多数 LLM 默认没有 pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Set pad_token to eos_token.")

    # 确保分词器具有 BOS 和 EOS token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
        print("Added bos_token to tokenizer.")

    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        print("Added eos_token to tokenizer.")

    # 加载模型，启用 4-bit 量化以节省资源（需要安装 bitsandbytes）
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            load_in_4bit=False,  # 根据需要启用 4-bit 量化
            torch_dtype=torch.float16,  # 使用 float16 精度
            trust_remote_code=True
        ).cuda()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 调整模型的嵌入层以适应新的分词器
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.eval()

    # 打印模型和分词器配置信息以验证
    print(f"Model vocab size: {model.config.vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"BOS Token ID: {tokenizer.bos_token_id}")
    print(f"EOS Token ID: {tokenizer.eos_token_id}")
    print(f"Pad Token ID: {tokenizer.pad_token_id}")

    # 加载数据，确保 raw_text=False 以获取分词后的数据
    print("Loading dataset for inference...")
    inference_data_tokenized = get_dataset(data_split='test', args=args, raw_text=False)
    inference_data_raw = get_dataset(data_split='test', args=args, raw_text=True)
    
    ########## method str ###########
    method_str = f'{args.method}'
    if args.method == 'sample':
        method_str += f',top_p={args.top_p},temperature={args.temperature}'
        
    # 初始化生成输出
    generated_data = []
    print("Generating outputs...")
    for idx in tqdm(range(len(inference_data_tokenized)), desc="Generating outputs"):
        instance_tokenized = inference_data_tokenized[idx]
        instance_raw = inference_data_raw[idx]

        # 创建 question_tokens 的副本，避免修改原始数据
        question_tokens = instance_tokenized['question'].copy()

        # 插入 BOS token（如果存在）
        bos_token_id = tokenizer.bos_token_id
        # if bos_token_id is not None:
        #     question_tokens = [bos_token_id] + question_tokens  # 创建新列表，避免 in-place 修改

        # 准备输入张量
        prompt_tokens = torch.as_tensor(question_tokens).unsqueeze(0).long().cuda()  # 形状：[1, seq_len]
        print(prompt_tokens.shape)
        # 打印 Prompt Tokens（调试）
        # print(f"Prompt Tokens {idx+1}: {question_tokens}")  # 可选：取消注释以调试

        # 生成文本
        generated_text = generate(prompt_tokens, model, tokenizer, args)


        # 使用原始的 raw text 作为 prompt
        prompt_text = instance_raw['question']
        if isinstance(prompt_text, list):
            prompt_text = ''.join(prompt_text)

        # 添加生成结果到输出数据
        generated_data = inference_data_raw
        generated_data[idx]['generated_answer'] = generated_text[0] # save only the string instead of the list

        # 打印生成结果（调试）
        # print("!!!!!!!!!!!!!Decoded prompt is:"+ tokenizer.decode(question_tokens))
        print("!!!!!!!!!!!!!Decoded unsqueezed prompt is:"+ tokenizer.decode(prompt_tokens[0]))
        print(f"Generated {idx+1}/{len(inference_data_tokenized)}: {generated_text[0]}\n")

    # 保存生成的内容到文件
    os.makedirs(pjoin('ssg_llama3/alog', args.exp, 'gen'), exist_ok=True)
    with open(pjoin('ssg_llama3/alog', args.exp, 'gen', f'{method_str}.json'), 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, indent=2, ensure_ascii=False)

    print(f"Generation complete. Output saved to {pjoin('ssg_llama2/alog', args.exp, 'gen', f'{method_str}.json')}")

    # 测试单个示例（可选）
    test_idx = 0  # 选择一个测试索引
    if len(inference_data_tokenized) > test_idx:
        test_instance_tokenized = inference_data_tokenized[test_idx]
        test_instance_raw = inference_data_raw[test_idx]

        test_question_tokens = test_instance_tokenized['question'].copy()
        # if bos_token_id is not None:
        #     test_question_tokens = [bos_token_id] + test_question_tokens

        test_prompt_tokens = torch.as_tensor(test_question_tokens).unsqueeze(0).long().cuda()

        test_generated_text = generate(test_prompt_tokens, model, tokenizer, args)

        test_prompt_text = test_instance_raw['question']
        if isinstance(test_prompt_text, list):
            test_prompt_text = ''.join(test_prompt_text)

        print(f"Test Prompt: {test_prompt_text}")
        print(f"Test Generated Answer: {test_generated_text}")
        
        # 在主函数末尾添加 pipeline 测试
    print("\nTesting with Hugging Face pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_with_pipeline(model, tokenizer, device)

if __name__ == "__main__":
    main()
