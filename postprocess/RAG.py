import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
import pandas as pd 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from tqdm import tqdm 
import json
from openai import OpenAI
import numpy as np
from postprocess.prompts import prompt1, prompt2, prompt3, prompt4, prompt5
import importlib
import torch, gc

class RAGPostProcessor:
    def __init__(self, configs, train_data_path, logger):
        self.train_data_path = train_data_path
        self.llm_name = configs['llm_name']
        self.threshold = configs['threshold']
        self.topk = configs['topk']
        self.api_key = configs['api_key']
        self.api_base = configs['api_base']
        self.persist_directory = configs['persist_directory']
        self.device = configs['device']
        self.prompt = self._import_prompt(configs['prompt'])
        self.logger = logger

    def _import_prompt(self, prompt_name):
        # 动态导入 postprocess.prompts 模块
        module = importlib.import_module('postprocess.prompts')
        # 获取对应的 prompt 对象
        return getattr(module, prompt_name)
    def get_llm(self, llm_name):
        if "gpt" in llm_name:
            llm = ChatOpenAI(model_name=llm_name, temperature=0)
        else:
            llm = self.get_local_llm(llm_name)
        return llm

    def ask_ChatGPT(self, prompt_content):
        client = OpenAI(
            base_url = self.api_base,
            api_key  = self.api_key,
        )
        chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user", 
                        "content": prompt_content
                    }
                ],
                model="gpt-3.5-turbo",
                temperature=0,
            )
        content = chat_completion.choices[0].message.content
        return content 


    def get_vectordb(self, normal_log_entries):
        # 做成Embedding存入到Vector db
        embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
        # 读取训练数据
        if not os.path.exists(self.persist_directory):
            os.mkdir(self.persist_directory)
            
        if not os.listdir(self.persist_directory):
            # 保存到硬盘
            self.logger.info('Using embedding...')
            vectordb = Chroma.from_texts(
                texts=normal_log_entries,
                embedding=embedding,
                persist_directory=self.persist_directory
            )
        else:
            # 从硬盘load
            self.logger.info('Loading from db...')
            if self.prompt == "prompt1" or self.prompt == "prompt2":
                self.persist_directory = "db/none"
            
            vectordb = Chroma(
                persist_directory=self.persist_directory, 
                embedding_function=embedding
            )
        return vectordb
    
    def get_retriever(self, retriever_type='thr', vectordb='None'):
        if retriever_type == "mmr":
            retriever = vectordb.as_retriever(
                        search_type="mmr", 
                        search_kwargs={"k": 5}
                    )
        elif retriever_type == "thr":
            if self.prompt == "prompt3":
                self.topk = 1
            else:
                self.topk = 5
            retriever = vectordb.as_retriever(
                        search_type="similarity_score_threshold", 
                        search_kwargs={"score_threshold": self.threshold, "k":self.topk}
                    )
        return retriever

    def get_local_llm(self, model_path):
        
        self.logger.info(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # model = AutoModelForCausalLM.from_pretrained(model_path)

        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            # do_sample=True,
            temperature=0,
            repetition_penalty=1.1,
            pad_token_id=2,
            return_full_text=True,
            max_new_tokens=1000,
        )

        mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        return mistral_llm


    def get_normal_log_entries(self):
        # get normal log entries from self.train_data_path
        train_df = pd.read_csv(self.train_data_path)
        train_df = train_df[train_df['Label'] == '-']
        normal_log_entries = train_df['EventTemplate'].unique().tolist()
        return normal_log_entries

    def post_process(self, anomaly_logs_path, test_data_path):
        result_path = 'output/anomaly_logs_detc_by_rag.csv'        
        answer_path = 'output/llm_answer.json'
        
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template=self.prompt)
        normal_log_entries = self.get_normal_log_entries()
        self.logger.info(f"Normal log templates to embedding: , {len(normal_log_entries)}")
        vector_db = self.get_vectordb(normal_log_entries)
        retriever = self.get_retriever("thr", vector_db)
        qa_chain = RetrievalQA.from_chain_type(
                        self.get_llm(self.llm_name),
                        chain_type='stuff',
                        retriever= retriever,
                        return_source_documents=True,
                        # verbose=True,
                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                    )
        pos_df = pd.read_csv(anomaly_logs_path)
        test_df = pd.read_csv(test_data_path)
 
        # 将Template按照数量从多到少排序，先看数据量多的
        pos_event_template_counts = pos_df['EventTemplate'].value_counts().to_dict()
        test_event_template_counts = test_df['EventTemplate'].value_counts().to_dict()
        pos_log_templates  = list(pos_event_template_counts.keys())

        df_result = pd.DataFrame(columns=['is_anomaly', f'frequency_inpos', 'frequency_intest', 'EventTemplate', 'reason', 'topk_similary_log_list'])

        answer_list = []
        
        for test_log in tqdm(pos_log_templates):

            answer = qa_chain.invoke({"query": test_log})
            # 提取每个Document对象的page_content
            answer['source_documents'] = [doc.page_content for doc in answer['source_documents']]
            
            topk_similary_log_list = answer['source_documents']
            answer_list.append(answer)
            
            content = answer['result']
            try:
                result = json.loads(content)
            except Exception as e:
                self.logger.info('---begin---')
                self.logger.info(e)
                self.logger.info(content)
                self.logger.info('---end-----')
                prompt = "Please keep only the Json part of the following content, and fill the is_anomaly into the \'is_anomaly\', \
                fill the reason into the \'reason\' field of the json. The returned content only needs a string in json format, Input:\n\n"
                prompt_content = prompt + content      
                content = self.ask_ChatGPT(prompt_content)
                self.logger.info("regenerate: "  + content)
                result = json.loads(content)
            # 提取is_anomaly和reason
            is_anomaly = result['is_anomaly']
            try:
                reason = result['reason']
            except:
                reason = "None"
            # 将数据追加到DataFrame
            df_result = pd.concat([ df_result, 
                                    pd.DataFrame({
                                        'is_anomaly':[is_anomaly],
                                        'frequency_inpos':[int(pos_event_template_counts[test_log])],
                                        'frequency_intest':[int(test_event_template_counts[test_log])],
                                        'EventTemplate': [test_log], 
                                        'reason': [reason],
                                        'topk_similary_log_list': [topk_similary_log_list]
                                        })
                                ], ignore_index=True)
            
            df_result.to_csv(result_path, index=False)
        df_result.to_csv(result_path, index=False)

        with open(answer_path, 'w') as file:
            json.dump(answer_list, file)
            
        self.logger.info(f'Saved results to \n{result_path}\n{answer_path}')
        
        anomaly_templates = df_result[df_result['is_anomaly'] == 1]['EventTemplate']
        filtered_df = pos_df[pos_df['EventTemplate'].isin(anomaly_templates)]
        return filtered_df['LineId'].tolist()