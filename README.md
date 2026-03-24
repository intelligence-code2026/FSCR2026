# 细粒度类案检索系统

本项目基于大语言模型（LLM）实现细粒度类案检索，通过关键情节抽取与匹配，提升类案检索的精确性与可解释性，并支持从检索结果中生成裁判规则。

## 整体流程

1. 采用 LLM 进行细粒度类案检索，主要流程包括：
   - 抽取查询案例和候选案例的关键情节
   - 生成关键情节的文本向量
   - 计算余弦相似度
   - 按照相似度阈值进行关键情节匹配

2. 为提高关键情节抽取质量，采用 DSPy 优化抽取关键情节的提示词；同时使用 Optuna 优化相似度阈值。

3. 为解决细粒度类案检索缺乏数据集的难题，本项目采用合成案例构建数据集，合成方法如下：
   - 使用提示词引导 LLM 抽取每个候选案例的关键情节，随机选取若干关键情节，由 LLM 合成一个完整的案例，合成的案例包括选取的全部关键情节
   - 由另外一个 LLM 检查合成案例的质量，不符合要求的案例将重新合成

4. 合成案例集合被分为训练集、验证集、测试集：
   - 训练集用于优化抽取关键情节的提示词
   - 验证集用于优化相似度阈值
   - 测试集不参与优化，用于检验模型性能

5. 优化后的细粒度类案检索模型可应用于真实案例的检索，并根据检索结果生成裁判规则

上述过程均由 `main_program/FSCR_llm_g1.py` 完成。

## 基线模型

在 `main_program` 中包括两个基线模型：
- `baseline_1.py`：采用全文文本向量计算测试集中合成案例（查询案例）与候选案例的余弦相似度，进行类案检索
- `baseline_2.py`：采用 BM25 算法计算测试集中合成案例（查询案例）与候选案例的相似度，进行类案检索

## 对比模型

`main_program/comparison_test.py` 用于实现采用 LLM 模拟人工判断进行细粒度类案检索，作为对比模型。

## 真实案例相似度分析

`main_program/similarity_query_candidate.py` 用于计算真实查询案例与候选案例全文余弦相似度的分布。

## 泛化能力测试

另选取 35 条真实案例作为候选案例，生成合成案例构成补充测试集合，采用 `FSCR_llm_g1.py` 优化后的模型进行细粒度类案检索，计算 Precision、Recall、F1-score，以检验模型的泛化能力。  
上述过程由 `supplementary_test_set/llm_g1/supplementary_test_program.py` 和`supplementary_test_set/llm_g2/supplementary_test_program.py`实现，分别对应第1组和第2组LLM。

## 不同 LLM 的对比实验

为对比不同 LLM 对计算结果的影响，采用另外一组 LLM 实现 `FSCR_llm_g1.py` 的过程，该实验程序位于文件夹 `FSCR_llm_g2` 中的 `FSCR_llm_g2.py`。

## 项目结构
├── main_program/
│   ├── FSCR_llm_g1.py          # 主流程：优化、训练、评估（第1组LLM）
│   ├── baseline_1.py           # 基线模型：全文向量检索
│   ├── baseline_2.py           # 基线模型：BM25检索
│   ├── comparison_test.py      # 对比模型：LLM模拟人工判断
│   └── similarity_query_candidate.py  # 真实案例全文相似度分布分析
│
├── supplementary_test_set/
│   ├── llm_g1/
│   │   └── supplementary_test_program.py   # 第1组LLM泛化能力测试
│   └── llm_g2/
│       └── supplementary_test_program.py   # 第2组LLM泛化能力测试
│
├── FSCR_llm_g2/
     └── FSCR_llm_g2.py          # 第2组LLM的主流程实现

## 说明

- 本项目所有流程均在上述文件中实现，各模块功能与文中描述一致
- 合成案例、阈值优化、提示词优化等关键环节均按所述方法执行
- api_key.xlsx保存的是LLM的API keys
- 本项目采用 custom non-commercial license，详细信息见 LICENSE 文件。

# Fine-Grained Similar Case Retrieval System

This project implements fine-grained similar case retrieval based on Large Language Models (LLMs). By extracting and matching key circumstances, it improves the accuracy and interpretability of case retrieval and supports the generation of adjudication patterns from retrieval results.

## Overall Workflow

1. LLMs are used for fine-grained similar case retrieval. The main workflow includes:
   - Extracting key circumstances from the query case and candidate cases
   - Generating text embeddings for the key circumstances
   - Calculating cosine similarity
   - Matching key circumstances according to the similarity threshold

2. To improve the quality of key circumstances extraction, DSPy is used to optimize the prompt for extracting key circumstances. Meanwhile, Optuna is used to optimize the similarity threshold.

3. To address the challenge of lacking datasets for fine-grained similar case retrieval, this project constructs datasets using synthetic cases. The synthesis method is as follows:
   - The prompt is used to guide the LLM to extract key circumstances from each candidate case. Several key circumstances are randomly selected, and the LLM synthesizes a complete case that includes all selected key circumstances
   - Another LLM checks the quality of the synthesized cases. Cases that do not meet the requirements are re-synthesized

4. The synthetic case set is divided into training, validation, and test sets:
   - The training set is used to optimize the prompts for extracting key circumstances
   - The validation set is used to optimize the similarity threshold
   - The test set is not involved in optimization and is used to evaluate model performance

5. The optimized fine-grained similar case retrieval model can be applied to real case retrieval and generate adjudication patterns based on retrieval results

The above processes are all completed by `main_program/FSCR_llm_g1.py`.

## Baseline Models

The `main_program` includes two baseline models:
- `baseline_1.py`: Uses full-text embeddings to calculate cosine similarity between synthetic cases (query cases) in the test set and candidate cases  for case retrieval
- `baseline_2.py`: Uses the BM25 algorithm to calculate similarity between synthetic cases (query cases) in the test set and candidate cases for case retrieval

## Comparison Model

`main_program/comparison_test.py` implements fine-grained similar case retrieval using LLMs to simulate human judgment as a comparison model.

## Real Case Similarity Analysis

`main_program/similarity_query_candidate.py` calculates the distribution of full-text cosine similarity between real query cases and candidate cases.

## Generalization Ability Test

Additionally, 35 real cases are selected as candidate cases, and synthetic cases are generated to form a supplementary test set. The optimized model from `FSCR_llm_g1.py` is used for fine-grained similar case retrieval, and Precision, Recall, and F1-score are calculated to evaluate the model's generalization ability.  
The above process is implemented by `supplementary_test_set/llm_g1/supplementary_test_program.py` and `supplementary_test_set/llm_g2/supplementary_test_program.py`, corresponding to the 1st group and the 2nd group of LLMs, respectively.


## Comparison Experiment with Different LLMs

To compare the impact of different LLMs on the results, another set of LLMs is used to implement the process of `FSCR_llm_g1.py`. This experimental program is located in `FSCR_llm_g2/FSCR_llm_g2.py`.

## Project Structure
├── main_program/
│ ├── FSCR_llm_g1.py # Main workflow: optimization, training, evaluation
│ ├── baseline_1.py # Baseline model: full-text embedding retrieval
│ ├── baseline_2.py # Baseline model: BM25 retrieval
│ ├── comparison_test.py # Comparison model: LLM-simulated human judgment
│ └── similarity_query_candidate.py # Real case full-text similarity distribution analysis
├── supplementary_test_set/
│   ├── llm_g1/
│   │   └── supplementary_test_program.py   # Generalization ability test of the 1st group of LLMs
│   └── llm_g2/
│       └── supplementary_test_program.py   # Generalization ability test of the 2nd group of LLMs
└── FSCR_llm_g2/
     └── FSCR_llm_g2.py # Main workflow implementation with the 2nd group of LLMs

## Notes

- All processes of this project are implemented in the above files, and the functions of each module are consistent with the descriptions in the text
- Key steps such as synthetic case generation, threshold optimization, and prompt optimization are all performed according to the described methods
- `api_key.xlsx` stores the API keys for LLMs
- This project adopts a custom non-commercial license. For details, please refer to the LICENSE file.