import json
from operator import itemgetter
import os
import glob

# 创建输出目录
os.makedirs('outputs_wrong', exist_ok=True)

# 获取所有JSON文件
json_files = glob.glob('outputs/*.json')

# 处理每个JSON文件
for outputs_path in json_files:
    # 获取文件名用于创建子目录
    base_name = os.path.splitext(os.path.basename(outputs_path))[0]
    output_dir = os.path.join('outputs_wrong', base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    with open(outputs_path, 'r', encoding='utf-8') as f:
        evaluate_results = json.load(f)

    # 处理每个问题
    for problem_idx, result in enumerate(evaluate_results):
        if False in result['accuracy']:  # 只处理包含False的问题
            
            question = result['question'] if 'question' in result else result['Problem'] if 'Problem' in result else result['problem']
            tokens = result['tokens']
            correct_answer = result['Solution'] if 'Solution' in result else result['answer'] if 'answer' in result else result['solution']
            
            # 处理每个错误答案
            for answer_idx, (acc, ans) in enumerate(zip(result['accuracy'], result['responses'])):
                if not acc:
                    # 创建文件名：问题索引:04d - 答案编号
                    file_name = f"{problem_idx:04d}-{answer_idx+1}.txt"
                    file_path = os.path.join(output_dir, file_name)
                    
                    # 写入问题、token数量、标准答案、预测答案和错误答案到单独的文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Tokens: {tokens}\n\n")
                        f.write(f"Question:\n{question}\n\n")
                        f.write(f"Correct Answer:\n{correct_answer}\n\n")
                        f.write(f"Gold:\n{result['gold']}\n\n")
                        # 添加预测答案（如果存在）
                        if 'prediction' in result:
                            f.write(f"Wrong Prediction:\n{result['prediction'][answer_idx]}\n\n")
                        f.write(f"Wrong Answer:\n{ans}\n")