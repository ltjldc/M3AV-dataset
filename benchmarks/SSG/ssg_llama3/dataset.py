import json
import os
from os.path import join as pjoin
from copy import deepcopy as c
from itertools import chain
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Load the set split information
set_split = json.load(open('ssg_dataset/sample_set_split.json', encoding='utf-8'))

# Specify model and cache directory
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/data/milsrg1/huggingface/cache/tl578/cache"

# Load tokenizer and add padding token if necessary
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

# def get_dataset(data_split, args, raw_text=False):
#     vid_list = sorted([i for i, j in set_split.items() if j == data_split])
#     if args.paper or args.subset:
#         vid_list = [i for i in vid_list if i.startswith('CHI') or i.startswith('Ubi')]

#     # Define prompts
#     if not args.paper:
#         p2l_question_prompt = [
#             "# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a slideshow creator and generate the text in the corresponding single slide based on the given speech text.\n# Speech text:\n",
#             "\n# Slide text:\n"
#         ]
#         l2p_question_prompt = [
#             "# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text in the given single slide.\n# Slide text:\n",
#             "\n# Speech text:\n"
#         ]
#     else:
#         p2l_question_prompt = [
#             "# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a slideshow creator and generate the text in the corresponding single slide based on the given speech text and related sentences in the paper.\n# Speech text:\n",
#             "\n# Related sentences in the paper:\n",
#             "\n# Slide text:\n"
#         ]
#         l2p_question_prompt = [
#             "# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text in the given single slide and related sentences in the paper.\n# Slide text:\n",
#             "\n# Related sentences in the paper:\n",
#             "\n# Speech text:\n"
#         ]

#     dialog_list = []

#     for vid in tqdm(vid_list, desc='process data', mininterval=10):
#         with open(f'ssg_dataset/text_data/{data_split}/{vid}.json', 'r', encoding='utf-8') as f:
#             seg_speech_ocr = json.load(f)

#         for unit_index, unit in enumerate(seg_speech_ocr):
#             if raw_text:
#                 process_fn = lambda sentence: c(sentence)
#             else:
#                 # Tokenize the sentence without adding special tokens
#                 process_fn = lambda sentence: tokenizer.encode(sentence, add_special_tokens=False)
#             speech_tokens = process_fn(unit['speech_text'])
#             ocr_tokens = process_fn(unit['ocr_text'])
#             if args.paper:
#                 paper_tokens = process_fn(unit['paper_text'])

#             if args.task == 'p2l':
#                 question = (
#                     process_fn(p2l_question_prompt[0]) +
#                     speech_tokens +
#                     process_fn(p2l_question_prompt[1])
#                 )
#                 if args.paper:
#                     question += paper_tokens + process_fn(p2l_question_prompt[2])
#                 answer = ocr_tokens
#             elif args.task == 'l2p':
#                 question = (
#                     process_fn(l2p_question_prompt[0]) +
#                     ocr_tokens +
#                     process_fn(l2p_question_prompt[1])
#                 )
#                 if args.paper:
#                     question += paper_tokens + process_fn(l2p_question_prompt[2])
#                 answer = speech_tokens
#             else:
#                 raise NotImplementedError

#             dialog_list.append({
#                 'question': question,
#                 'answer': answer
#             })

#     return dialog_list
def get_dataset(data_split, args, raw_text=False):
    vid_list = sorted([i for i, j in set_split.items() if j == data_split])
    if args.paper or args.subset:
        vid_list = [i for i in vid_list if i.startswith('CHI') or i.startswith('Ubi')]

    # Define single continuous user template
    user_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>{}{}\n# Speech text:\n<|eot_id|>"

    # Define prompts
    if not args.paper:
        p2l_question_prompt = [
            "# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a slideshow creator and generate the text in the corresponding single slide based on the given speech text.\n# Speech text:\n",
            "\n# Slide text:\n"
        ]
        l2p_question_prompt = [
            "There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text in the given single slide.\n# Slide text:\n",
            "\n# Speech text:\n"
        ]
    else:
        p2l_question_prompt = [
            "# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a slideshow creator and generate the text in the corresponding single slide based on the given speech text and related sentences in the paper.\n# Speech text:\n",
            "\n# Related sentences in the paper:\n",
            "\n# Slide text:\n"
        ]
        l2p_question_prompt = [
            "# There is a single slide and the speaker's speech within a video clip. The clip is a part of the whole speech video. Please act like a speaker and generate the corresponding speech text based on the text in the given single slide and related sentences in the paper.\n# Slide text:\n",
            "\n# Related sentences in the paper:\n",
            "\n# Speech text:\n"
        ]

    dialog_list = []

    for vid in tqdm(vid_list, desc='process data', mininterval=10):
        with open(f'ssg_dataset/text_data/{data_split}/{vid}.json', 'r', encoding='utf-8') as f:
            seg_speech_ocr = json.load(f)

        for unit_index, unit in enumerate(seg_speech_ocr):
            if raw_text:
                process_fn = lambda sentence: c(sentence)
            else:
                # Tokenize the sentence without adding special tokens
                process_fn = lambda sentence: tokenizer.encode(sentence, add_special_tokens=False)
            speech_tokens = process_fn(unit['speech_text'])
            ocr_tokens = process_fn(unit['ocr_text'])
            if args.paper:
                paper_tokens = process_fn(unit['paper_text'])

            # Convert tokens to text for ocr/speech content and construct question text
            if args.task == 'p2l':
                ocr_text = " ".join(unit['ocr_text']) if raw_text else tokenizer.decode(ocr_tokens)
                # prompt_text = p2l_question_prompt[0]
                # if args.paper:
                #     prompt_text += p2l_question_prompt[1] + p2l_question_prompt[2]
                # question_text = user_template.format(prompt_text, ocr_text)
                # question = process_fn(question_text)
                # answer = ocr_tokens

            elif args.task == 'l2p':
                speech_text = "".join(unit['speech_text']) if raw_text else tokenizer.decode(speech_tokens)
                ocr_text = "".join(unit['ocr_text']) if raw_text else tokenizer.decode(ocr_tokens)
                prompt_text = l2p_question_prompt[0]
                if args.paper:
                    prompt_text += l2p_question_prompt[1] + l2p_question_prompt[2]
                question_text = user_template.format(prompt_text, ocr_text)
                question = process_fn(question_text)
                answer = speech_tokens

            else:
                raise NotImplementedError

            dialog_list.append({
                'question': question,
                'answer': answer
            })

    return dialog_list


def build_input_from_segments(instance, args):
    """Build a sequence of input from the question and answer."""
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    # Add BOS and EOS tokens if they exist
    if bos_token_id is not None:
        instance['question'].insert(0, bos_token_id)
    if eos_token_id is not None:
        instance['answer'].append(eos_token_id)

    # Concatenate question and answer
    instance["input_ids"] = list(chain(instance['question'], instance['answer']))

    # Create labels for language modeling
    instance["lm_labels"] = (
        [-100] * len(instance['question']) + instance['answer']
    )

    return instance

class AVSDDataSet(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        instance = self.data[index]
        instance = build_input_from_segments(instance, self.args)
        input_ids = torch.as_tensor(instance["input_ids"]).long()
        lm_labels = torch.as_tensor(instance["lm_labels"]).long()
        return input_ids, lm_labels

def collate_batch(batch, args):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        result = torch.ones((len(seq), max_len)).long() * pad_token
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result

    input_ids_list, lm_labels_list = zip(*batch)

    pad_token_id = tokenizer.pad_token_id

    input_ids = padding(input_ids_list, pad_token_id)
    lm_labels = padding(lm_labels_list, -100)
    input_mask = input_ids != pad_token_id

    return input_ids, lm_labels, input_mask

if __name__ == '__main__':
    from args import get_args
    _, args = get_args('train')
    print(args)

    train_dataset = AVSDDataSet(
        data=get_dataset('dev', args, raw_text=False),
        args=args
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=3,
                              shuffle=True,
                              collate_fn=lambda x: collate_batch(x, args))
    for batch in tqdm(train_loader):
        print([tokenizer.decode(i.tolist(), skip_special_tokens=True) for i in batch[0]])
        print(batch[1])
        print(batch[2])
        break
