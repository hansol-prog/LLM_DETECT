import json 
import pickle 
import numpy as np
import random 
from tqdm import tqdm

# 입력: 글 1개에 대한 문장, 형태소, 품사 정보
# Input: Sentence, Morpheme, and Part-of-Speech Information for a Single Text
def analyze_comma_usage(sentences, morphs, pos):

    results = {}

    # 글 1건에서 사용된 총 쉼표 개수 
    # Total Number of Commas Used in a Single Text
    total_comma_count_per_text = 0

    # 글 1건을 구성하는 문장들에서 사용된 쉼표 개수의 평균
    # Average Number of Commas Used per Sentence in a Single Text
    num_commas_per_sentence = []

    # 글 1건에서 쉼표를 포함하고 있는 문장의 비율
    # Proportion of Sentences Containing Commas in a Single Text
    num_sentences = len(sentences)
    num_comma_include_sentences = 0

    # 글 1건을 구성하는 문장들에서 쉼표가 사용된 위치의 상대적 위치
    # Relative Position of Commas Used in Sentences in a Single Text
    relative_positions_per_sentence = []
    
    # 글 1건을 구성하는 문장들에서 쉼표로 나뉜 분절 길이
    # Length of Segments Divided by Commas in Sentences in a Single Text
    segment_lengths_per_sentence = []
    
    # 글 1건을 구성하는 문장들에서 쉼표 앞뒤 형태소 품사
    # Part-of-Speech of Morphemes Before and After Commas in Sentences in a Single Text
    pos_patterns_per_sentence = []

    # 글 1건을 구성하는 문장들에서 쉼표 앞뒤 형태소 품사 다양성 점수
    # Diversity Score of Part-of-Speech of Morphemes Before and After Commas in Sentences in a Single Text
    pos_patterns_diversity_score_per_sentence = []
    
    # 글 1건을 구성하는 문장들에서 쉼표 사용 비율
    # Rate of Comma Usage in Sentences in a Single Text
    comma_usage_rate_per_sentence = []
    
    # 글 1건에 대하여 쉼표가 사용된 위치의 평균값 및 표준편차
    # Average and Standard Deviation of Relative Position of Commas Used in a Single Text
    avg_relative_position_per_sentence = []
    std_relative_position_per_sentence = []
    
    # 글 1건에 대하여 쉼표로 나뉜 문장 분절 길이의 평균값 및 표준편차
    # Average and Standard Deviation of Length of Segments Divided by Commas in a Single Text
    avg_segment_length_per_sentence = []
    std_segment_length_per_sentence = []

    for morp, p in zip(morphs, pos):

        if ',' in morp:
            num_comma_include_sentences += 1

        commas = [i for i, m in enumerate(morp) if m == ',']
        num_commas = len(commas) 
        total_comma_count_per_text += num_commas
        num_commas_per_sentence.append(num_commas)

        if num_commas > 0: 
            relative_positions = [comma / len(morp) for comma in commas]
            relative_positions_per_sentence.append(relative_positions)
            # 쉼표로 잘린 분절 길이 (쉼표는 길이 계산에서 제외)
            # Length of segments divided by commas (excluding commas in length calculation)
            segment_lengths = [len(morp[start+1:end]) if start in commas else len(morp[start:end]) 
                            for start, end in zip([0] + commas, commas + [len(morp)])]
            segment_lengths_per_sentence.append(segment_lengths)
            # 쉼표 앞뒤 형태소 품사
            # Part-of-speech of morphemes before and after commas
            pos_patterns = [(p[i-1], p[i+1]) for i in commas if 0 < i < len(p)-1] 
            pos_patterns_per_sentence.append(pos_patterns)

            # 쉼표 앞뒤 형태소 품사 다양성 점수
            # Diversity score of part-of-speech of morphemes before and after commas
            if len(pos_patterns) == 0:
                pos_patterns_diversity_score = 0
            else:
                pos_patterns_diversity_score = len(set(pos_patterns)) / len(pos_patterns)   
            pos_patterns_diversity_score_per_sentence.append(pos_patterns_diversity_score)

            num_morphs = len(morp)
            comma_usage_rate_per_sentence.append(num_commas / num_morphs)

            avg_relative_position = np.mean(relative_positions)
            std_relative_position = np.std(relative_positions)
            avg_segment_length = np.mean(segment_lengths)
            std_segment_length = np.std(segment_lengths)

            avg_relative_position_per_sentence.append(avg_relative_position)
            std_relative_position_per_sentence.append(std_relative_position)
            avg_segment_length_per_sentence.append(avg_segment_length)
            std_segment_length_per_sentence.append(std_segment_length)

        else: 
            relative_positions_per_sentence.append([])
            segment_lengths_per_sentence.append([])
            pos_patterns_per_sentence.append([])
            comma_usage_rate_per_sentence.append(0)

            # 쉼표 앞뒤 형태소 품사 다양성 점수
            # Diversity score of part-of-speech of morphemes before and after commas
            pos_patterns_diversity_score_per_sentence.append(0)

            avg_relative_position_per_sentence.append(0)
            std_relative_position_per_sentence.append(0)
            avg_segment_length_per_sentence.append(0)
            std_segment_length_per_sentence.append(0)

    # 글 1건에 대하여 쉼표를 포함하고 있는 문장의 비율 / 쉼표를 포함하고 있는 문장 수를 전체 문장 수로 나눈 값
    # Proportion of Sentences Containing Commas in a Single Text
    comma_include_rate_per_text = num_comma_include_sentences / num_sentences
    results['comma_include_sentence_rate_per_text'] = comma_include_rate_per_text

    # 글 1건에 대하여 쉼표를 포함하고 있는 문장의 수
    # Number of Sentences Containing Commas in a Single Text
    results['num_comma_include_sentences_per_text'] = num_comma_include_sentences

    # 글 1건에서 사용된 총 쉼표 개수 
    # Total Number of Commas Used in a Single Text
    results['total_comma_count_per_text'] = total_comma_count_per_text

    # 글 1건을 구성하는 문장들에서 전체 형태소 개수 대비 쉼표 개수의 비율의 평균값
    # Average Rate of Commas Used per Sentence in a Single Text
    avg_comma_usage_rate_per_text = np.mean(comma_usage_rate_per_sentence)
    results['avg_comma_usage_rate_per_text'] = avg_comma_usage_rate_per_text

    # 글 1건에 대하여 쉼표가 사용된 위치의 평균값
    # Average Relative Position of Commas Used in a Single Text
    avg_relative_position_per_text = np.mean(avg_relative_position_per_sentence)
    results['avg_relative_position_per_text'] = avg_relative_position_per_text
    # 글 1건에 대하여 쉼표가 사용된 위치의 표준편차
    # Standard Deviation of Relative Position of Commas Used in a Single Text
    std_relative_position_per_text = np.std(avg_relative_position_per_sentence)
    results['std_relative_position_per_text'] = std_relative_position_per_text

    # 글 1건에 대하여 쉼표로 나뉜 문장 분절 길이의 평균값
    # Average Length of Segments Divided by Commas in a Single Text
    avg_segment_length_per_text = np.mean(avg_segment_length_per_sentence)
    results['avg_segment_length_per_text'] = avg_segment_length_per_text
    # 글 1건에 대하여 쉼표로 나뉜 문장 분절 길이의 표준편차
    # Standard Deviation of Length of Segments Divided by Commas in a Single Text
    std_segment_length_per_text = np.std(avg_segment_length_per_sentence)
    results['std_segment_length_per_text'] = std_segment_length_per_text

    # 글 1건에 대하여 각 문장별 쉼표 앞뒤 형태소 품사 다양성 점수
    # Diversity Score of Part-of-Speech of Morphemes Before and After Commas in Sentences in a Single Text
    avg_pos_patterns_diversity_score_per_text = np.mean(pos_patterns_diversity_score_per_sentence)
    results['avg_pos_patterns_diversity_score_per_text'] = avg_pos_patterns_diversity_score_per_text

    return results

def analyze_pos_ngram_diversity(pos):
    pos_1grams_per_sentence = []
    pos_2grams_per_sentence = []
    pos_3grams_per_sentence = []
    pos_4grams_per_sentence = []
    pos_5grams_per_sentence = []

    pos_1grams_diversity_per_sentence = []
    pos_2grams_diversity_per_sentence = []
    pos_3grams_diversity_per_sentence = []
    pos_4grams_diversity_per_sentence = []
    pos_5grams_diversity_per_sentence = []

    pos_num_1grams_per_text = []
    pos_num_2grams_per_text = []
    pos_num_3grams_per_text = []
    pos_num_4grams_per_text = []
    pos_num_5grams_per_text = []

    for p in pos:
        unigrams = p
        bigrams = [(p[i-1], p[i]) for i in range(1, len(p))]
        trigrams = [(p[i-2], p[i-1], p[i]) for i in range(2, len(p))]
        fourgrams = [(p[i-3], p[i-2], p[i-1], p[i]) for i in range(3, len(p))]
        fivegrams = [(p[i-4], p[i-3], p[i-2], p[i-1], p[i]) for i in range(4, len(p))]

        pos_1grams_per_sentence.append(unigrams)
        pos_2grams_per_sentence.append(bigrams)
        pos_3grams_per_sentence.append(trigrams)
        pos_4grams_per_sentence.append(fourgrams)
        pos_5grams_per_sentence.append(fivegrams)

        pos_num_1grams_per_text.append(len(unigrams))
        pos_num_2grams_per_text.append(len(bigrams))
        pos_num_3grams_per_text.append(len(trigrams))
        pos_num_4grams_per_text.append(len(fourgrams))
        pos_num_5grams_per_text.append(len(fivegrams))

        pos_1grams_diversity_per_sentence.append(len(set(unigrams)) / len(unigrams) if len(unigrams) > 0 else 0)
        pos_2grams_diversity_per_sentence.append(len(set(bigrams)) / len(bigrams) if len(bigrams) > 0 else 0)
        pos_3grams_diversity_per_sentence.append(len(set(trigrams)) / len(trigrams) if len(trigrams) > 0 else 0)
        pos_4grams_diversity_per_sentence.append(len(set(fourgrams)) / len(fourgrams) if len(fourgrams) > 0 else 0)
        pos_5grams_diversity_per_sentence.append(len(set(fivegrams)) / len(fivegrams) if len(fivegrams) > 0 else 0)

    # 글 1건에서 사용된 총 POS 1-gram 개수 / 2-gram 개수 / 3-gram 개수 / 4-gram 개수 / 5-gram 개수
    # Total Number of POS 1-gram / 2-gram / 3-gram / 4-gram / 5-gram Used in a Single Text
    results = {}
    results['pos_num_1grams_per_text'] = sum(pos_num_1grams_per_text)
    results['pos_num_2grams_per_text'] = sum(pos_num_2grams_per_text)
    results['pos_num_3grams_per_text'] = sum(pos_num_3grams_per_text)
    results['pos_num_4grams_per_text'] = sum(pos_num_4grams_per_text)
    results['pos_num_5grams_per_text'] = sum(pos_num_5grams_per_text)

    # 글 1건을 구성하는 문장들에서 사용된 POS 1-gram 개수의 평균값 / 2-gram 개수의 평균값 / 3-gram 개수의 평균값 / 4-gram 개수의 평균값 / 5-gram 개수의 평균값
    # Average Number of POS 1-gram / 2-gram / 3-gram / 4-gram / 5-gram Used per Sentence in a Single Text
    results['avg_pos_num_1grams_per_text'] = np.mean(pos_num_1grams_per_text)
    results['avg_pos_num_2grams_per_text'] = np.mean(pos_num_2grams_per_text)
    results['avg_pos_num_3grams_per_text'] = np.mean(pos_num_3grams_per_text)
    results['avg_pos_num_4grams_per_text'] = np.mean(pos_num_4grams_per_text)
    results['avg_pos_num_5grams_per_text'] = np.mean(pos_num_5grams_per_text)

    # 글 1건을 구성하는 문장들에서 사용된 POS 1-gram 개수의 표준편차 / 2-gram 개수의 표준편차 / 3-gram 개수의 표준편차 / 4-gram 개수의 표준편차 / 5-gram 개수의 표준편차
    # Standard Deviation of Number of POS 1-gram / 2-gram / 3-gram / 4-gram / 5-gram Used per Sentence in a Single Text
    results['std_pos_num_1grams_per_text'] = np.std(pos_num_1grams_per_text)
    results['std_pos_num_2grams_per_text'] = np.std(pos_num_2grams_per_text)
    results['std_pos_num_3grams_per_text'] = np.std(pos_num_3grams_per_text)
    results['std_pos_num_4grams_per_text'] = np.std(pos_num_4grams_per_text)
    results['std_pos_num_5grams_per_text'] = np.std(pos_num_5grams_per_text)

    # 글 1건을 구성하는 문장들에서 POS 1-gram 다양성 점수의 평균값
    # Average POS 1-gram Diversity Score per Sentence in a Single Text
    # POS 1-gram 다양성 점수 = 고유한 POS 1-gram 수 / 전체 POS 1-gram 수
    # POS 1-gram Diversity Score = Number of Unique POS 1-grams / Total Number of POS 1-grams
    avg_pos_1grams_diversity_per_text = np.mean(pos_1grams_diversity_per_sentence)
    results['avg_pos_1grams_diversity_per_text'] = avg_pos_1grams_diversity_per_text
    avg_pos_2grams_diversity_per_text = np.mean(pos_2grams_diversity_per_sentence)
    results['avg_pos_2grams_diversity_per_text'] = avg_pos_2grams_diversity_per_text
    avg_pos_3grams_diversity_per_text = np.mean(pos_3grams_diversity_per_sentence)
    results['avg_pos_3grams_diversity_per_text'] = avg_pos_3grams_diversity_per_text
    avg_pos_4grams_diversity_per_text = np.mean(pos_4grams_diversity_per_sentence)
    results['avg_pos_4grams_diversity_per_text'] = avg_pos_4grams_diversity_per_text
    avg_pos_5grams_diversity_per_text = np.mean(pos_5grams_diversity_per_sentence)
    results['avg_pos_5grams_diversity_per_text'] = avg_pos_5grams_diversity_per_text

    # 글 1건을 구성하는 문장들에서 POS 1-gram 다양성 점수의 표준편차
    # Standard Deviation of POS 1-gram Diversity Score per Sentence in a Single Text
    std_pos_1grams_diversity_per_text = np.std(pos_1grams_diversity_per_sentence)
    results['std_pos_1grams_diversity_per_text'] = std_pos_1grams_diversity_per_text
    std_pos_2grams_diversity_per_text = np.std(pos_2grams_diversity_per_sentence)
    results['std_pos_2grams_diversity_per_text'] = std_pos_2grams_diversity_per_text
    std_pos_3grams_diversity_per_text = np.std(pos_3grams_diversity_per_sentence)
    results['std_pos_3grams_diversity_per_text'] = std_pos_3grams_diversity_per_text
    std_pos_4grams_diversity_per_text = np.std(pos_4grams_diversity_per_sentence)
    results['std_pos_4grams_diversity_per_text'] = std_pos_4grams_diversity_per_text
    std_pos_5grams_diversity_per_text = np.std(pos_5grams_diversity_per_sentence)
    results['std_pos_5grams_diversity_per_text'] = std_pos_5grams_diversity_per_text

    return results

if __name__ == '__main__':
    # Load the data
    with open(f"train_pos_taging_results.pkl", "rb") as f:
        sentence_level_ana = pickle.load(f)
    with open('./data/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f) # .loads()가 아닌 .load()를 사용합니다.

    # Get the data
    human_texts = []
    llm_texts = []
    for inst in data: 
        if inst['generated'] == '0': 
            human_texts.append(inst['full_text'])
        else: 
            llm_texts.append(inst['full_text'])


    human_sentences = sentence_level_ana['human']['sentences']
    human_morphs = sentence_level_ana['human']['morphs']
    human_pos = sentence_level_ana['human']['pos']
    llm_sentences = sentence_level_ana['llm']['sentences']
    llm_morphs = sentence_level_ana['llm']['morphs']
    llm_pos = sentence_level_ana['llm']['pos']

    # Feature Analysis
    human_comma_ana = []
    llm_comma_ana = []
    human_pos_ngram_ana = []
    llm_pos_ngram_ana = []
    for human_s, human_m, human_p in tqdm(zip(human_sentences, human_morphs, human_pos), desc="Analyzing Human Texts"):
        human_comma_ana.append(analyze_comma_usage(human_s, human_m, human_p))
        human_pos_ngram_ana.append(analyze_pos_ngram_diversity(human_p))
    for llm_s, llm_m, llm_p in tqdm(zip(llm_sentences, llm_morphs, llm_pos),desc="Analyzing LLM Texts"):
        llm_comma_ana.append(analyze_comma_usage(llm_s, llm_m, llm_p))
        llm_pos_ngram_ana.append(analyze_pos_ngram_diversity(llm_p))

    human_comma_include_sentence_rate_per_text = [ana['comma_include_sentence_rate_per_text'] for ana in human_comma_ana]
    human_num_comma_include_sentences_per_text = [ana['num_comma_include_sentences_per_text'] for ana in human_comma_ana]
    human_total_comma_count_per_text = [ana['total_comma_count_per_text'] for ana in human_comma_ana]
    human_avg_comma_usage_rate_per_text = [ana['avg_comma_usage_rate_per_text'] for ana in human_comma_ana]
    human_avg_relative_position_per_text = [ana['avg_relative_position_per_text'] for ana in human_comma_ana]
    human_std_relative_position_per_text = [ana['std_relative_position_per_text'] for ana in human_comma_ana]
    human_avg_segment_length_per_text = [ana['avg_segment_length_per_text'] for ana in human_comma_ana]
    human_std_segment_length_per_text = [ana['std_segment_length_per_text'] for ana in human_comma_ana]
    human_pos_diversity_score_before_after_comma_per_text = [ana['avg_pos_patterns_diversity_score_per_text'] for ana in human_comma_ana]
    llm_comma_include_sentence_rate_per_text = [ana['comma_include_sentence_rate_per_text'] for ana in llm_comma_ana]
    llm_num_comma_include_sentences_per_text = [ana['num_comma_include_sentences_per_text'] for ana in llm_comma_ana]
    llm_total_comma_count_per_text = [ana['total_comma_count_per_text'] for ana in llm_comma_ana]
    llm_avg_comma_usage_rate_per_text = [ana['avg_comma_usage_rate_per_text'] for ana in llm_comma_ana]
    llm_avg_relative_position_per_text = [ana['avg_relative_position_per_text'] for ana in llm_comma_ana]
    llm_std_relative_position_per_text = [ana['std_relative_position_per_text'] for ana in llm_comma_ana]
    llm_avg_segment_length_per_text = [ana['avg_segment_length_per_text'] for ana in llm_comma_ana]
    llm_std_segment_length_per_text = [ana['std_segment_length_per_text'] for ana in llm_comma_ana]
    llm_pos_diversity_score_before_after_comma_per_text = [ana['avg_pos_patterns_diversity_score_per_text'] for ana in llm_comma_ana]

    # Construct data for ML experiment
    human_features = []
    llm_features = []
    for include, usage, position, segment, pos_diversity in zip(human_comma_include_sentence_rate_per_text, human_avg_comma_usage_rate_per_text, human_avg_relative_position_per_text, human_avg_segment_length_per_text, human_pos_diversity_score_before_after_comma_per_text):
        human_features.append([include, usage, position, segment, pos_diversity])
    for include, usage, position, segment, pos_diversity in zip(llm_comma_include_sentence_rate_per_text, llm_avg_comma_usage_rate_per_text, llm_avg_relative_position_per_text, llm_avg_segment_length_per_text, llm_pos_diversity_score_before_after_comma_per_text):
        llm_features.append([include, usage, position, segment, pos_diversity])

    human_ml_data = []
    llm_ml_data = []
    for text, feature in zip(human_texts, human_features):
        item = {}
        item['text'] = text
        item['feature'] = feature
        item['label'] = 0
        item['written_by'] = 'Human'
        human_ml_data.append(item)
    for text, feature in zip(llm_texts, llm_features):
        item = {}
        item['text'] = text
        item['feature'] = feature
        item['label'] = 1
        item['written_by'] = 'LLM'
        llm_ml_data.append(item)

    # Data Split
    # human_ml_data를 seed를 사용하여 8:2로 random 하게 자르기 
    # Randomly divide human_ml_data into 8:2 using seed
    random.seed(42)
    random.shuffle(human_ml_data)
    train_size = int(len(human_ml_data) * 0.8)
    train_human_ml_data = human_ml_data[:train_size]

    # Save the data
    ml_data = {} 
    ml_data['train'] = train_human_ml_data + llm_ml_data 

    with open('data/train_ml_data.pkl', 'wb') as f: 
        pickle.dump(ml_data, f)