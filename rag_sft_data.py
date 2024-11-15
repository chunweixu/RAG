# 导入所需的库
from datasets import load_dataset
import anthropic
import random
import os
from http import HTTPStatus
import dashscope
import json
from collections import defaultdict

dashscope.api_key
# 全局常量定义
DATA_PATH = "./data/MultiHopRAG.json"
TRANSLATION_PROMPT = "Translate the following text into Chinese. Retain all names, company names, and other proper nouns in the original language. Here is the text:"

# 添加输出路径常量
OUTPUT_DIR = "./data"
PROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "processed_data.json")
FINETUNING_DATA_PATH = os.path.join(OUTPUT_DIR, "finetuning_data.json")
TRAIN_DATA_PATH = os.path.join(OUTPUT_DIR, "train_data.json")
EVAL_DATA_PATH = os.path.join(OUTPUT_DIR, "eval_data.json")

def save_json(data, filepath):
    """保存数据到JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# def translate_text(text, model="claude-3-5-sonnet-20241022", max_tokens=1024):
#     """
#     使用Anthropic的Claude模型进行翻译。

#     参数:
#     - text (str): 要翻译的文本。
#     - api_key (str): Anthropic API密钥。
#     - model (str): 使用的Claude模型名称。
#     - max_tokens (int): 生成的最大token数量。
    
#     返回:
#     - str: Claude模型的翻译结果。
#     """
#     client = anthropic.Anthropic(api_key)
    
#     # 调用Claude模型进行翻译
#     message = client.messages.create(
#         model=model,
#         max_tokens=max_tokens,
#         messages=[
#             {"role": "user", "content": text}
#         ]
#     )

#     return message.content[0].text

def translate_text(content):
    messages = [{'role': 'user', 'content': content}]
    responses = dashscope.Generation.call(
        "qwen-max",
        messages=messages,
        result_format='message',  # set the result to be "message" format.
        stream=True,  # set streaming output
        incremental_output=True  # get streaming output incrementally
    )
    
    response_content = ""  # 初始化一个空字符串来收集输出
    
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            output_text = response.output.choices[0]['message']['content']
            response_content += output_text  # 将内容追加到response_content中
        else:
            error_message = (
                'Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message)
            )
             # 记录错误信息到日志
            return content
            print(error_message)
    
    # logger.info(f"API call completed with content: {response_content}")  # 记录成功信息到日志
    return response_content

# 定义翻译的prompt，确保人名和专有名词保留原文
translation_prompt = "Translate the following text into Chinese. Retain all names, company names, and other proper nouns in the original language. Here is the text:"

# 定义函数来处理 DatasetDict 数据结构
def process_dataset(dataset):
    processed_data = []

    for i in range(len(dataset)):
        item = dataset[i]
        # 1. 翻译 query 和 answer 字段
        translated_query = translate_text(f"{translation_prompt} {item['query']}")
        translated_answer = translate_text(f"{translation_prompt} {item['answer']}")

        # 2. 保留 question_type 原文
        question_type = item['question_type']

        # 3. 处理 evidence_list，仅保留每个item的 fact 字段并进行翻译
        translated_evidence_list = [
            translate_text(f"{translation_prompt} {evidence['fact']}")
            for evidence in item['evidence_list']
        ]

        # 4. 生成新的数据结构
        processed_item = {
            'query': translated_query,
            'answer': translated_answer,
            'question_type': question_type,
            'evidence_list': translated_evidence_list
        }

        # 添加到处理后的数据列表
        processed_data.append(processed_item)

    return processed_data

def prepare_rag_finetuning_data(dataset):
    finetuning_data = []

    for item in dataset:
        # 生成优化后的 Prompt
        query = item['query']
        evidence_facts = " ".join(item['evidence_list'])
        prompt = (
        f"""任务：根据提供的背景信息准确回答下列问题。\n\n
        背景信息："{evidence_facts}" \n\n
        问题："{query}"\n\n
        回答要求：\n
        - 使用背景信息进行回答。如背景信息中没有明确答案，请简要说明并向用户表达歉意。\n
        - 回答应尽量简洁准确，限制在2-3句话内。\n
        - 如回答涉及多个要点，请分条列出，保持清晰明了。\n
        - 针对直接事实型问题，务必引用背景信息中的内容，避免主观推断。\n
        - 如果背景信息不足以解答，请表明无法解答，并建议用户提供更多相关信息。\n\n
        回答："""
    )


        # 生成 Response：直接使用 Answer 字段
        response = item['answer']

        # 保留 Question_Type 字段
        question_type = item['question_type']

        # 创建新结构并加入结果列表
        finetuning_item = {
            'Prompt': prompt,
            'Response': response,
            'Question_Type': question_type
        }
        finetuning_data.append(finetuning_item)

    return finetuning_data
    
import random
from collections import defaultdict

def create_train_eval_split(dataset, eval_size=50):
    # 统计每种 question_type 的数量
    type_to_items = defaultdict(list)
    for item in dataset:
        question_type = item['Question_Type']
        type_to_items[question_type].append(item)

    # 计算每种类型在评测集中的样本数
    total_types = len(type_to_items)
    samples_per_type = max(1, eval_size // total_types)

    # 从每种类型中抽取样本，确保评测集的多样性
    eval_set = []
    for question_type, items in type_to_items.items():
        # 如果某类型样本少于指定数量，全部放入评测集
        if len(items) <= samples_per_type:
            eval_samples = items
        else:
            eval_samples = random.sample(items, samples_per_type)
        
        eval_set.extend(eval_samples)

    # 如果评测集不足50条，从剩余数据中继续随机抽样补足
    eval_set = eval_set[:eval_size]  # 保证评测集不超过 eval_size 条
    train_set = [item for item in dataset if item not in eval_set]

    return train_set, eval_set    


def main():
    # 1. 读取数据
    dataset = load_dataset('json', data_files=DATA_PATH)
     # 测试模式：只取前5条数据
    test_dataset = dataset['train']
    
    # 2. 处理数据集并保存
    processed_dataset = process_dataset(test_dataset)
    save_json(processed_dataset, PROCESSED_DATA_PATH)
    print(f"已保存处理后的数据集到: {PROCESSED_DATA_PATH}")
    
    # 3. 准备 RAG 微调数据集并保存
    finetuning_dataset = prepare_rag_finetuning_data(processed_dataset)
    save_json(finetuning_dataset, FINETUNING_DATA_PATH)
    print(f"已保存微调数据集到: {FINETUNING_DATA_PATH}")
    
    # 4. 创建训练集和评估集划分并保存
    train_set, eval_set = create_train_eval_split(finetuning_dataset)
    save_json(train_set, TRAIN_DATA_PATH)
    save_json(eval_set, EVAL_DATA_PATH)
    print(f"已保存训练集到: {TRAIN_DATA_PATH}")
    print(f"已保存评估集到: {EVAL_DATA_PATH}")
    
    return train_set, eval_set

if __name__ == "__main__":
    main()
