import json
import torch
from transformers import AutoTokenizer, AutoModel, set_seed
from torch.nn.functional import cosine_similarity

SEED_VALUE = 42
set_seed(SEED_VALUE)

# 配置缓存目录
CACHE_DIR = "/data/milsrg1/huggingface/cache/tl578/cache"

# 加载 LLaMA 模型和分词器
llama_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, cache_dir=CACHE_DIR)
llama_model = AutoModel.from_pretrained(llama_model_name, cache_dir=CACHE_DIR)
llama_model.eval()

# 加载 Jina 模型和分词器
jina_model_name = "ISOISS/jina-embeddings-v3-tei"
jina_tokenizer = AutoTokenizer.from_pretrained(jina_model_name, cache_dir=CACHE_DIR)
jina_model = AutoModel.from_pretrained(jina_model_name, cache_dir=CACHE_DIR)
jina_model.eval()

# 从中间结果文件读取提取的信息单元
intermediate_output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_info_units_jina.json'
with open(intermediate_output_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取 "extraction" 部分作为 extracted_units
extracted_units = []
for item in data:
    extraction = item.get("extraction", {})
    extracted_units.append({
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "generated_answer": item.get("generated_answer", ""),
        "extraction": {
            "Script A Information Units": extraction.get("Script A Information Units", []),
            "Script B Information Units": extraction.get("Script B Information Units", [])
        }
    })

# Stage 2: 信息单元匹配并计算相似度
matching_results = []
for units in extracted_units:
    answer_units = units["extraction"]["Script A Information Units"]
    generated_units = units["extraction"]["Script B Information Units"]

    # 初始化软匹配指标
    tp = 0
    unique_b_score_sum = 0
    matched_a_units = {}

    # 编码 Script A 和 Script B 的所有信息单元
    embeddings_a = []
    embeddings_b = []

    for a_unit in answer_units:
        inputs = jina_tokenizer(a_unit, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = jina_model(**inputs)
            embeddings_a.append(output.last_hidden_state.mean(dim=1))

    for b_unit in generated_units:
        inputs = jina_tokenizer(b_unit, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = jina_model(**inputs)
            embeddings_b.append(output.last_hidden_state.mean(dim=1))

    # 计算 Script B 的 TP 和 soft unique 分数
    for i, b_embedding in enumerate(embeddings_b):
        max_score, best_match_idx = 0, None

        for j, a_embedding in enumerate(embeddings_a):
            # 计算余弦相似度
            similarity = cosine_similarity(b_embedding, a_embedding).item()
            print(f"Cosine Similarity for B Unit '{generated_units[i]}' vs A Unit '{answer_units[j]}' - Score: {similarity:.4f}")

            if similarity > max_score:
                max_score = similarity
                best_match_idx = j

        tp += max_score
        unique_b_score_sum += (1 - max_score)
        if best_match_idx is not None:
            matched_a_units[best_match_idx] = max(matched_a_units.get(best_match_idx, 0), max_score)

    # 计算 Script A 的 soft unique 值
    unique_a_score_sum = sum(1 - score for score in matched_a_units.values())

    # 计算精确率、召回率和 F1 分数
    fp = unique_b_score_sum
    fn = unique_a_score_sum

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 记录结果
    matching_results.append({
        "question": units["question"],
        "answer": units["answer"],
        "generated_answer": units["generated_answer"],
        "extraction": units["extraction"],
        "metrics": {
            "total_cosine_tp": round(tp, 2),
            "unique_b_score_sum": round(unique_b_score_sum, 2),
            "unique_a_score_sum": round(unique_a_score_sum, 2),
            "Precision": round(precision, 2),
            "Recall": round(recall, 2),
            "F1 Score": round(f1_score, 2)
        }
    })
    print(f"Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}, "
          f"Total Cosine TP: {tp:.2f}, Unique B Score Sum: {unique_b_score_sum:.2f}, Unique A Score Sum: {unique_a_score_sum:.2f}")

# 保存最终结果
output_file = '/research/milsrg1/user_workspace/tl578/M3AV-dataset/benchmarks/SSG/ssg_llama3/alog/l2p/gen/llama3_jina_eval.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(matching_results, f, indent=2, ensure_ascii=False)

print(f"Final evaluation results saved to {output_file}")
