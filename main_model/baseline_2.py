'''本程序采用BM25算法进行全文类案检索'''
'''This program performs full-text similar case retrieval using the BM25 algorithm.'''

import pickle
from rank_bm25 import BM25Okapi
import jieba

def tokenize(text):
    return [w for w in jieba.cut(text) if w.strip() and w not in {"是", "的", "了", "在", "和", "与", "或", "及", "等", "个", "种", 
                 "。", "，", "：", "；", "、", "（", "）", "“", "”"}]

dict_case_text_pickle=r'.\data\dspy_opted_model\dict_case_text.pkl'
with open(dict_case_text_pickle,'rb') as f_obj9:
     dict_case_text=pickle.load(f_obj9)  
     
texts=[dict_case_text[key] for key in dict_case_text.keys()]   #候选案例文本构成的列表
tokenized_texts = [tokenize(text) for text in texts]
bm25 = BM25Okapi(tokenized_texts)
    

list_query_test_pickle=r'.\data\dspy_opted_model\list_query_test.pkl'
with open(list_query_test_pickle,'rb') as f_obj12:
     list_query_test=pickle.load(f_obj12)  
     
hit_quyery=0
precision_sum=0
recall_sum=0
f1_sum=0
for key_query in range(len(list_query_test)):
    sorted_list=[]
    tokens_test = tokenize(list_query_test[key_query]['text'])
    scores = bm25.get_scores(tokens_test)
    list_sorted_list_code = (-scores).argsort()[:3]  # top-3
    for i in list(list_sorted_list_code):
        for key_text in dict_case_text.keys():
            if texts[i]== dict_case_text[key_text]:
               sorted_list.append(key_text) 
        
    label_key=list(list_query_test[key_query]['label'].keys())
    intersection = list(set(sorted_list) & set(label_key))
    if len(intersection) !=0:
        hit_quyery+=1
    precision_query=len(intersection)/len(sorted_list)
    recall_query = len(intersection) / len(label_key) 
    if precision_query + recall_query > 0:
        f1 = 2 * (precision_query * recall_query) / (precision_query + recall_query)
    else:
        f1 = 0
    precision_sum+=precision_query
    recall_sum+=recall_query
    f1_sum+=f1
        
hit_at_3=hit_quyery/len(list_query_test)
precision_total=precision_sum/len(list_query_test)
recall_total=recall_sum/len(list_query_test)
f1_score = f1_sum/len(list_query_test)
print(f'\n查询案例共{len(list_query_test)}条，候选案例共{len(dict_case_text)}条，求得:\nhit@3={hit_at_3}，\nprecision@3={precision_total}，\nrecall@3={recall_total}，\nf1_score@3={f1_score}')
        
    