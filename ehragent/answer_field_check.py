import json
import os

from dotenv import load_dotenv
load_dotenv()

for data in ['mimic_iii', 'eicu']:
    answer_book = f"{os.getenv('DATASET_PATH')}/{data}/valid_preprocessed.json"

    contents = []
    with open(answer_book, 'r') as f:
        # for line in f:
        #     contents.append(json.loads(line))
        contents = json.load(f)

    items_with_answers = []
    items_without_answers = []

    for item in contents:
        if 'answer' in item:
            items_with_answers.append(item['id'])
        else:
            items_without_answers.append(item['id'])

    print(f"For Dataset: {data}")
    print(f"Total items: {len(contents)}")
    print(f"Items WITH 'answer' field: {len(items_with_answers)}")
    print(f"Items WITHOUT 'answer' field: {len(items_without_answers)}")