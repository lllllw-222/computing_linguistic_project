for corrupt_rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    CUDA_VISIBLE_DEVICES=2 python strqa_inf_factual_rating.py --dola_layers high --output_path strategyqa/result/2.2.2_cot --no_comment --cot --corrupt_rate $corrupt_rate
done