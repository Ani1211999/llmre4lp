import argparse
from transformers import AutoTokenizer
import torch

import os
import json

import ray
import sys
import os.path as osp
sys.path.append("/mnt/webscistorage/cc7738/ws_aniket/llmre4lp/src/LLM")

from fastchat.model.LLM4HeG import LLM4HeGForCausalLM
from fastchat.conversation import SeparatorStyle, get_conv_template
from fastchat.utils import disable_torch_init

from tqdm import tqdm

def load_eval_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_eval_file(args.eval_file)

    if args.end_id == -1:
        args.end_id = len(prompt_file)
    else:
        args.end_id += 1

    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))

    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')
    
    os.makedirs(args.output_res_path, exist_ok=True)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        # eval_model(
        #         args, prompt_file[start_idx:end_idx], start_split, end_split
        #     )
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    disable_torch_init()
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print('finish loading')

    print('start loading')
    model = LLM4HeGForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    # print("1:{}".format(torch.cuda.memory_allocated(0)))

    res_data = []
    print(f'total: {len(prompt_file)}')
    
    for instruct_item in tqdm(prompt_file):
        conv = get_conv_template("llm4heg")
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        # if args.eval_file.endswith("long_range_infer.json"):
        #     # Modified prompt to enforce probability output
        #     prompt_text = (f"Background: Papers in the arXiv 2023 network. Each paper is described by its title and abstract. "
        #                    f"Task: Predict the probability (0 to 1) that a citation relationship exists between two papers based on their descriptions. "
        #                    f"Paper 1 description: {instruct_item['conversations'][0]['value'].split('Paper 1 description: ')[1].split('Paper 2 description: ')[0]}"
        #                    f"Paper 2 description: {instruct_item['conversations'][0]['value'].split('Paper 2 description: ')[1].split('. Answer template:')[0]}"
        #                    f". Provide only a numeric probability value between 0 and 1 (e.g., 0.75), without any additional text.")
        # else:
        # for sentence in instruct_item["conversations"][:-1]:
        #     role = roles[sentence["from"]]
        #     conv.append_message(role, sentence["value"])
        raw_text = instruct_item["conversations"][0]["value"]
        paper1 = raw_text.split("Paper 1 description: ")[1].split("Paper 2 description: ")[0].strip()
        paper2 = raw_text.split("Paper 2 description: ")[1].split(". Answer template:")[0].strip()

        # Construct a clearer, inference-time prompt
        inference_prompt = (
            "Background: Papers in the arXiv 2023 network. Each paper is described by its title and abstract.\n"
            "Task: Predict the probability (0 to 1) that a citation relationship exists between two papers based on their descriptions.\n\n"
            f"Paper 1 description:\n{paper1}\n\n"
            f"Paper 2 description:\n{paper2}\n\n"
            "Answer format: Only provide a numeric probability between 0 and 1 (e.g., 0.72). Do not explain or justify your answer."
        )

        # Use chat format with 1 user message
        conv.append_message(roles["human"], inference_prompt)
        conv.append_message(roles["gpt"], None)
      
        prompt = conv.get_prompt()
        print(prompt)
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.ADD_COLON_TWO else conv.sep2

        # print("2:{}".format(torch.cuda.memory_allocated(0)))

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024
            )

        # print("3:{}".format(torch.cuda.memory_allocated(0)))

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        res_data.append({"id": instruct_item["id"], "res": outputs}.copy())
        with open(osp.join(args.output_res_path, 'preds_lr.json'), "w") as fout:    
            json.dump(res_data, fout, indent=4)

        # print("4:{}".format(torch.cuda.memory_allocated(0)))

        del input_ids
        torch.cuda.empty_cache()

        # print("5:{}".format(torch.cuda.memory_allocated(0)))

    return res_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=-1)

    args = parser.parse_args()
    ray.init()
    run_eval(args, args.num_gpus)