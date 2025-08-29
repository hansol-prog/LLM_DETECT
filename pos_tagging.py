from konlpy.tag import Kkma
import kss
from konlpy.utils import pprint
import json 
import pickle 
import argparse

if __name__ == '__main__':
    # POS tagger
    kkma = Kkma()

    # Load data
    with open('./data/train.json', 'r', encoding='utf-8') as f:
        data = json.load(f) # .loads()가 아닌 .load()를 사용합니다.
    human_texts = []
    llm_texts = []
    for inst in data:
        if inst['generated'] == '0': 
            human_texts.append(inst['full_text'])
        else: 
            llm_texts.append(inst['full_text'])

    # Sentence segmentation
    human_sentences = [] 
    llm_sentences = []
    for text in human_texts:
        tmp_list = []
        kss_sentences = kss.split_sentences(text)
        for sentence in kss_sentences:
            tmp_list.extend(sentence.split('\n'))
        human_sentences.append(tmp_list)
    for text in llm_texts:
        tmp_list = []
        kss_sentences = kss.split_sentences(text)
        for sentence in kss_sentences:
            tmp_list.extend(sentence.split('\n'))
        llm_sentences.append(tmp_list)

    # POS tagging
    human_sentences_morphs = [] 
    llm_sentences_morphs = []
    human_sentences_pos = [] 
    llm_sentences_pos = []
    for sentences in human_sentences:
        tmp_morph = []
        tmp_pos = []
        for sentence in sentences:
            ana = kkma.pos(sentence)
            morph = []
            pos = [] 
            for item in ana:
                morph.append(item[0])
                pos.append(item[1])
            tmp_morph.append(morph)
            tmp_pos.append(pos)
        human_sentences_morphs.append(tmp_morph)
        human_sentences_pos.append(tmp_pos)
    for sentences in llm_sentences:
        tmp_morph = []
        tmp_pos = []
        for sentence in sentences:
            ana = kkma.pos(sentence)
            morph = []
            pos = [] 
            for item in ana:
                morph.append(item[0])
                pos.append(item[1])
            tmp_morph.append(morph)
            tmp_pos.append(pos)
        llm_sentences_morphs.append(tmp_morph)
        llm_sentences_pos.append(tmp_pos)

    # Save data
    total_ana = {} 
    human_ana = {} 
    llm_ana = {}
    human_ana['sentences'] = human_sentences
    human_ana['morphs'] = human_sentences_morphs
    human_ana['pos'] = human_sentences_pos
    llm_ana['sentences'] = llm_sentences
    llm_ana['morphs'] = llm_sentences_morphs
    llm_ana['pos'] = llm_sentences_pos
    total_ana['human'] = human_ana
    total_ana['llm'] = llm_ana

    with open(f'train_pos_taging_results.pkl', 'wb') as f:
        pickle.dump(total_ana, f)