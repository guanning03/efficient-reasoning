from utils.parser import extract_answer
from utils.grader import math_equal

DATASET_KEYS = {
    'openai/gsm8k': {'question': 'question', 'answer': 'answer'},
    'hendrycks/competition_math': {'question': 'problem', 'answer': 'solution'},
    'datasets/converted_aime_dataset': {'question': 'problem', 'answer': 'solution'},
    'di-zhang-fdu/MATH500': {'question': 'problem', 'answer': 'solution'},
    'datasets/compression_dataset': {'question': 'problem', 'answer': 'solution'},
    'Maxwell-Jia/AIME_2024': {'question': 'Problem', 'answer': 'Solution'},  
    'opencompass/AIME2025': {'question': 'question', 'answer': 'answer'}   
}

RESPONSE_EXTRACTOR = {
    'openai/gsm8k': lambda x: extract_answer(x, data_name='gsm8k'),
    'hendrycks/competition_math': lambda x: extract_answer(x, data_name='math'),
    'di-zhang-fdu/MATH500': lambda x: extract_answer(x, data_name='math'),
    'datasets/compression_dataset': lambda x: extract_answer(x, data_name='math'),
    'datasets/converted_aime_dataset': lambda x: extract_answer(x, data_name='math'),
    'Maxwell-Jia/AIME_2024': lambda x: extract_answer(x, data_name='math'),  # 补充
    'opencompass/AIME2025': lambda x: extract_answer(x, data_name='math')    # 补充
}

RESPONSE_COMPARATOR = {
    'openai/gsm8k': lambda x, y: math_equal(x, y, timeout=True),
    'hendrycks/competition_math': lambda x, y: math_equal(x, y, timeout=True),
    'di-zhang-fdu/MATH500': lambda x, y: math_equal(x, y, timeout=True),
    'datasets/compression_dataset': lambda x, y: math_equal(x, y, timeout=True),
    'datasets/converted_aime_dataset': lambda x, y: math_equal(x, y, timeout=True),
    'Maxwell-Jia/AIME_2024': lambda x, y: math_equal(x, y, timeout=True),  # 补充
    'opencompass/AIME2025': lambda x, y: math_equal(x, y, timeout=True)    # 补充
}

# def extract_official_dataset_id(dataset_name):
#     name = dataset_name.split('/')[-1]
#     dataset_dict = {
#         'gsm8k': 'openai/gsm8k',
#         'MATH500': 'di-zhang-fdu/MATH500',
#         'AIME_2024': 'Maxwell-Jia/AIME_2024',
#         'AIME2024': 'Maxwell-Jia/AIME_2024',
#         'AIME2025': 'opencompass/AIME2025',
#         'competition_math': 'hendrycks/competition_math',
#         'compression_dataset': 'datasets/compression_dataset',
#         'converted_aime_dataset': 'datasets/converted_aime_dataset'
#     }
#     return dataset_dict[name]


