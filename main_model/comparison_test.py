# 细粒度类案检索的对比实验（模拟人工进行细粒度检索）
# Comparative experiments on FSCR (simulating manual fine-grained search)

import openpyxl
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate)
import self_func
import re
import pickle
import numpy as np

xlsx_api_key = openpyxl.load_workbook(r'..\api_key.xlsx')
sheet_api_key = xlsx_api_key['Sheet1']
api_key_deepseek = sheet_api_key['B7'].value
api_key = sheet_api_key['B' + str(2)].value

system_template = "你是一个法律专家"
system_message_template = SystemMessagePromptTemplate.from_template(system_template)
output_parser=CommaSeparatedListOutputParser()  #实例化输
    
llm = ChatOpenAI(model='qwen-plus',base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=0.0,top_p=0.1,
                    openai_api_key=api_key)

llm_ds = ChatOpenAI(model='deepseek-chat',base_url="https://api.deepseek.com",
                    temperature=0.0,top_p=0.1,
                    openai_api_key=api_key_deepseek)

#加载生成的查询案例的合成数据
dict_query_pickle=r'.\data\dspy_opted_model\list_query_test.pkl'
with open(dict_query_pickle,'rb') as f_obj12:
      dict_query=pickle.load(f_obj12) 
        
     
file_path_comparation = r'.\data\prompt\comparation_text_prompt.txt'    #读取提示词文本文件
with open(file_path_comparation, 'r', encoding='utf-8') as file_comparation:       
      comparation_prompt = file_comparation.read()
human_message_prompt_comparation = HumanMessagePromptTemplate.from_template(comparation_prompt)
chat_prompt_comparation = ChatPromptTemplate.from_messages([system_message_template, human_message_prompt_comparation])
chain_comparation = chat_prompt_comparation|llm_ds|output_parser   # 新版langchain导入的链 

#加载候选案例的摘要
dict_summary_candidate_pickle=r'.\data\\dspy_opted_model\dict_candidate_summary.pkl'
with open(dict_summary_candidate_pickle, 'rb') as f_obj9:
    dict_summary_candidate = pickle.load(f_obj9) 

list_metrics=[]
list_metrics_2=[]

for query_key in  list(range(len(dict_query)))[:]:   #可修改加载的查询案例数量################################
    query_text= dict_query[query_key]['text'] 
    dict_candidate={}
    for candidate_key in list(dict_summary_candidate.keys()):
        candidate_summary=''
        summary_num=0
        for candidate_summary_key in list(dict_summary_candidate[candidate_key].keys())[1:]:
            candidate_summary+=f'{candidate_summary_key}.{dict_summary_candidate[candidate_key][candidate_summary_key]}。'
            summary_num+=1
        output_comparation_1=chain_comparation.invoke({'query_text':query_text,'candidate_summary':candidate_summary,'summary_num':summary_num})
        if '没有找到类似情节' not in output_comparation_1[0]:
            try:
                output_comparation = [int(x) for x in re.split(r'[;；]', output_comparation_1[0])]
                dict_candidate[candidate_key]=output_comparation
            except Exception as e:  # 使用 'except Exception as e' 来捕获异常
                print(f'不能生成关键情节索引列表的原因是{e}')
                print(f'当前输出是{output_comparation_1}')
                continue
        else:
            continue 
    if  dict_candidate!={}:                      
        # dict_query_case[query_key]= dict_candidate
        query_gold_dict=dict_query[query_key]['label']
        precision,recall,f1=self_func.metrics_func(dict_candidate, query_gold_dict)
        precision_2,recall_2,f1_2=self_func.metrics_func_2(dict_candidate, query_gold_dict)
        list_metrics.append([precision,recall,f1])
        list_metrics_2.append([precision_2,recall_2,f1_2]) 
        
    elif dict_candidate=={}:
        continue
   
list_metrics_array=np.array(list_metrics)
list_metrics_array_2=np.array(list_metrics_2)                  
metrics_mean=np.mean(list_metrics_array, axis=0)
metrics_mean_2=np.mean(list_metrics_array_2, axis=0)
print(f'对比实验的平均指标列表为{metrics_mean}。')
print(f'对比实验的第二个平均指标列表为{metrics_mean_2}。')
