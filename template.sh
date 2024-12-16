

CUDA_VISIBLE_DEVICES=0 python strqa_inf_factual_rating.py --output_path strategyqa/result/baseline --no_comment --cot --corrupt_rate 0.1

CUDA_VISIBLE_DEVICES=0 python strqa_inf_factual_rating.py --dola_layers high --output_path strategyqa/result/early_exit_layers
