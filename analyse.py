import json

normal_data = json.load(open('./result/2.1/Llama-3.1-8B_max_new_tokens-256_top_p-1.0_top_k-0_temperature-1.0_typical_p-1.0_repetition_penalty-1.0_do_sample-False_dola_layers-None_num_shots-6_cot-False_no_comment-False_do_shuffle-False_debug-False_seed-42/run_0.json'))

nocomment_data = json.load(open('./result/2.2.2_nocot/Llama-3.1-8B_max_new_tokens-256_top_p-1.0_top_k-0_temperature-1.0_typical_p-1.0_repetition_penalty-1.2_do_sample-False_dola_layers-low_num_shots-6_cot-False_no_comment-True_corrupt_rate-0.2_do_shuffle-False_debug-False_seed-42/run_0.json'))

# 统计normal data里有哪些题做对了，哪些题做错了。统计nocomment data里有哪些题做对了，哪些题做错了，哪些题是no comment的。然后比较两者的结果。
normal_right = set()
normal_wrong = set()
for i in range(len(normal_data['gold_answer'])):
    
    if normal_data['gold_answer'][i] == normal_data['model_answer'][i]:
        normal_right.add(i)
        
    else:
        normal_wrong.add(i)
        
nocomment_right = set()
nocomment_wrong = set()
nocomment_no_comment = set()
for i in range(len(nocomment_data['gold_answer'])):
    
    if nocomment_data['model_answer'][i] == 'no comment':
        nocomment_no_comment.add(i)
    
    elif nocomment_data['gold_answer'][i] == nocomment_data['model_answer'][i]:
        nocomment_right.add(i)
        
    else:
        nocomment_wrong.add(i)
        
# 计算normal data和nocomment data的正确率
normal_accuracy = len(normal_right) / len(normal_data['gold_answer'])
nocomment_accuracy = len(nocomment_right) / (len(nocomment_data['gold_answer']) - len(nocomment_no_comment))

# 计算进行了no comment操作后对模型表现的影响
# 1. no comment答案占总答案的比例
no_comment_rate = len(nocomment_no_comment) / len(nocomment_data['gold_answer'])
# 2. 原来做对的题里有多少被no comment了
right_no_comment = len(normal_right & nocomment_no_comment) / len(normal_right)
# 3. 原来做对的题有多少被改错了
right_wrong = len(normal_right & nocomment_wrong) / len(normal_right)
# 4. 原来做错的题有多少被no comment了
wrong_no_comment = len(normal_wrong & nocomment_no_comment) / len(normal_wrong)
# 5. 原来做错的题有多少被改对了
wrong_right = len(normal_wrong & nocomment_right) / len(normal_wrong)

print('normal accuracy:', normal_accuracy)
print('nocomment accuracy:', nocomment_accuracy)
print('no comment rate:', no_comment_rate)
print('right no comment:', right_no_comment)
print('right wrong:', right_wrong)
print('wrong no comment:', wrong_no_comment)
print('wrong right:', wrong_right)