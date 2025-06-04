export CUDA_VISIBLE_DEVICES=0

python eval_model.py --model_name_or_path ./vicuna_for_lp --eval_file ../../llm_pred/prompt_json/Arxiv/long_range_infer.json --output_res_path inference_results_lp_llm/long_range_edges/