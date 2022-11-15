from os import listdir
from os.path import isfile, join
import json
from keybert import KeyBERT
from transformers.pipelines import pipeline
from transformers import AutoModel, AutoTokenizer

DATASET_PATH = './news'
SECTION = "content"
NUM_OF_RESULT = 3

kw_model = KeyBERT()

def update_json_file(json_file, data):
    json_file.seek(0)
    json.dump(data, json_file, indent=4, ensure_ascii=False)
    return True

def run_it(file_name):
    print("On it: " + file_name)
    with open(DATASET_PATH+'/'+file_name, "r+", encoding='utf-8') as json_file:
        data = json.load(json_file)
        paper_content = data[SECTION]
        result = {}

        keywords = kw_model.extract_keywords(paper_content, top_n=NUM_OF_RESULT)
        result["default"] = keywords

        keywords = kw_model.extract_keywords(paper_content, use_mmr=True, diversity=0.5, top_n=NUM_OF_RESULT)
        result["default:diversed"] = keywords

        keywords = kw_model.extract_keywords(paper_content, keyphrase_ngram_range=(1, 2), top_n=NUM_OF_RESULT)
        result["ngram2"] = keywords

        keywords = kw_model.extract_keywords(paper_content, keyphrase_ngram_range=(1, 2), use_mmr=True, diversity=0.4, top_n=NUM_OF_RESULT)
        result["ngram2:diversed"] = keywords

        keywords = kw_model.extract_keywords(paper_content, keyphrase_ngram_range=(1, 3), top_n=NUM_OF_RESULT, use_maxsum= True)
        result["ngram3"] = keywords

        keywords = kw_model.extract_keywords(paper_content, keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.4, top_n=NUM_OF_RESULT)
        result["ngram3:diversed"] = keywords

        data["_result"] = result
        update_json_file(json_file, data)


only_files = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]

for file_name in only_files:
    run_it(file_name)