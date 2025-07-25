export CUDA_VISIBLE_DEVICES=0

python eval_scripts/eval_model_lr.py --model_name_or_path ./vicuna_for_lp_arxiv_subgraph --eval_file ../../llm_prompt_dataset/Arxiv/hop3_eval.json --output_res_path inference_results_lp_llm/Arxiv_3hop/test/