def metrics_func(dict_pred,dict_gold):
    if dict_pred=={}:
        return 0,0,0
    else:
        triples_pred={(x, y) for x in dict_pred for y in dict_pred[x]}
        triples_gold={(x, y) for x in dict_gold for y in dict_gold[x]}
        
        tp = len(triples_pred & triples_gold) 
        fp = len(triples_pred - triples_gold)
        fn = len(triples_gold - triples_pred)
    
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return precision,recall,f1

def metrics_func_2(dict_pred,dict_gold):
    if dict_pred=={}:
        return 0,0,0
    else:
        triples_pred=dict_pred.keys()
        triples_gold=dict_gold.keys()
        
        tp = len(triples_pred & triples_gold) 
        fp = len(triples_pred - triples_gold)
        fn = len(triples_gold - triples_pred)
    
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        return precision,recall,f1
    
def rank_by_max_similarity(embedding_dict_query,embedding_dict_candidate,num_candidate_selected):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    '''采用最大相似度平均法筛选与查询案例较为相似的候选案例的函数'''
    embedding_dict_query_temp=list(embedding_dict_query.values())[0]
    list_query=list(embedding_dict_query_temp.values())
    array_query=np.array(list_query)
    dict_sim={}
    for key in embedding_dict_candidate.keys():
        list_candidate=list(embedding_dict_candidate[key].values())
        array_candidate=np.array(list_candidate)
        qc_sim=cosine_similarity(array_query,array_candidate)
        value_sim=qc_sim.max(axis=1).mean()
        dict_sim[key]=value_sim
    sorted_keys_sim = sorted(dict_sim, key=dict_sim.get, reverse=True)
    return sorted_keys_sim[0:num_candidate_selected]

def rank_by_bidirectional_similarity(embedding_dict_query, embedding_dict_candidate, num_candidate_selected):
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    embedding_dict_query_temp = list(embedding_dict_query.values())[0]
    array_query = np.array(list(embedding_dict_query_temp.values()))
    dict_sim = {}
    
    for key in embedding_dict_candidate.keys():
        array_candidate = np.array(list(embedding_dict_candidate[key].values()))
        qc_sim = cosine_similarity(array_query, array_candidate)
        # print(f'相似度矩阵的形状是{qc_sim.shape}')
        # 正向匹配：query 每行找 candidate 中最相似的
        forward = qc_sim.max(axis=1).mean()
        # 反向匹配：candidate 每行找 query 中最相似的
        backward = qc_sim.max(axis=0).mean()

        # 综合相似度（可以调节权重 alpha）
        value_sim = 0.5 * forward + 0.5 * backward
        dict_sim[key] = value_sim

    sorted_keys_sim = sorted(dict_sim, key=dict_sim.get, reverse=True)
    return sorted_keys_sim[:num_candidate_selected]

