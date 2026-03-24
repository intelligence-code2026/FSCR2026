'''本程序采用全文词向量计算余弦相似度，进行类案检索。'''
'''This program performs similar case retrieval by calculating cosine similarity using full-text word vectors.'''

import openpyxl
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import pickle


xlsx_api_key = openpyxl.load_workbook(r'..\api_key.xlsx')
sheet_api_key = xlsx_api_key['Sheet1']
api_key_deepseek = sheet_api_key['B7'].value
api_key = sheet_api_key['B' + str(2)].value

client = OpenAI(
   api_key=sheet_api_key['B'+str(2)].value,  
   base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  )

def embedding_vector(sentence:str,client=client)->list:     
    completion = client.embeddings.create(
        model="text-embedding-v3",
        input=sentence,
        dimensions=1024,
        encoding_format="float"
    )
    return  completion.data[0].embedding

def similiar_score(embedding_vec_query, embedding_vec_candidate)->float:
    # 确保输入是numpy数组（如果你的输入已经是数组，可以省略这步）
    vec1 = np.array(embedding_vec_query).reshape(1, -1)  # 强制转为2D: (1, n)
    vec2 = np.array(embedding_vec_candidate).reshape(1, -1)  # 强制转为2D: (1, n)
    
    # 计算余弦相似度（结果是一个1x1矩阵）
    similarity_matrix = cosine_similarity(vec1, vec2)
    
    # 返回标量值（提取矩阵中的单个数值）
    return similarity_matrix[0][0]

dict_case_text_pickle=r'.\data\dspy_opted_model\dict_case_text.pkl'
with open(dict_case_text_pickle,'rb') as f_obj9:
     dict_case_text=pickle.load(f_obj9)  

list_query_test_pickle=r'.\data\dspy_opted_model\list_query_test.pkl'
with open(list_query_test_pickle,'rb') as f_obj12:
     list_query_test=pickle.load(f_obj12)  

#dict_embedding_candidate候选案例的词向量字典
dict_embedding_candidate={}
for key in dict_case_text.keys():
    dict_embedding_candidate[key]=embedding_vector(dict_case_text[key])

list_embedding_query=[]
for k in range(len(list_query_test)):
    embedding_test_text={}
    embedding_test_text['embedding']=embedding_vector(list_query_test[k]['text'])
    embedding_test_text['label']=list_query_test[k]['label']
    list_embedding_query.append(embedding_test_text)
    
dict_similiar_query={}
for key_querey in range(len(list_embedding_query)):    
    dict_similiar_candidate={}
    for key_candidate in dict_embedding_candidate.keys():
        dict_similiar_candidate[key_candidate]=similiar_score(list_embedding_query[key_querey]['embedding'], 
                                               dict_embedding_candidate[key_candidate])
    dict_similiar_query_temp={}
    dict_similiar_query_temp['similiar']=dict_similiar_candidate 
    dict_similiar_query_temp['label']= list_embedding_query[key_querey]['label']
    dict_similiar_query[key_querey]=dict_similiar_query_temp   
    
hit_quyery=0
precision_sum=0
recall_sum=0
f1_sum=0
for key_query in  dict_similiar_query.keys():
    sim_dict = dict_similiar_query[key_query]['similiar']
    sorted_list = sorted(sim_dict, key=sim_dict.get, reverse=True)[:3]
    label_key=list(dict_similiar_query[key_query]['label'].keys())
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
        
hit_at_3=hit_quyery/len(dict_similiar_query)
precision_total=precision_sum/len(dict_similiar_query)
recall_total=recall_sum/len(dict_similiar_query)
f1_score = f1_sum/len(dict_similiar_query)
print(f'\n查询案例共{len(dict_similiar_query)}条，候选案例共{len(dict_embedding_candidate)}条，求得:\nhit@3={hit_at_3}，\nprecision@3={precision_total}，\nrecall@3={recall_total}，\nf1_score@3={f1_score}')
        
    