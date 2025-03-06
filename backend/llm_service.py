import os
import json
from openai import OpenAIError
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings
import logging
import time
import traceback
import sys
import psutil  


class LLMService:

    def __init__(self):
        """
        Initialize LLM service by reading documents from local file
        """
        # set the path to save the vector store on local
        self.store_path = "vector_store"
        self.airtable_records = "airtable_records.txt"
        self.prompt_file = "prompts/system_prompts.json"

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables")
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large" 
            )
            self.vectorstore = self._initialize_vectorstore()
            self.llm = ChatOpenAI(temperature=0.8,
                                  model="gpt-4.5-preview",
                                  max_tokens=10000,
                                  frequency_penalty=0.4,
                                  presence_penalty=0.3,
                                  top_p=0.9)
            system_prompt = self._load_system_prompt()
            self.chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.vectorstore.as_retriever(),
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": system_prompt}
                if system_prompt else {},
                memory=None,
                chain_type="stuff",
                get_chat_history=lambda h: h)

        except Exception as e:
            print(f"Error in initializing llm service: {str(e)}")
            raise

    def _initialize_vectorstore(self):
        """
        Initialize vector store by loading existing one or creating a new one if it doesn't exist
        
        Returns:
            FAISS: The loaded or newly created vector store
        """
        # make sure the store path exists
        os.makedirs(self.store_path, exist_ok=True)
        if os.path.exists(os.path.join(self.store_path, "index.faiss")):
            print("Loading existing vector store...")
            return FAISS.load_local(self.store_path,
                                    self.embeddings,
                                    allow_dangerous_deserialization=True)
        else:
            print("Creating new vector store...")
            # todo: process linkedin and other documents to vector store
            return self.process_documents_to_vectorstore()

    def process_documents_to_vectorstore(self):
        """
        Process Airtable records from local file to vector store line by line
        
        Returns:
            FAISS: The vector store containing processed documents
        """
        try:
            if not os.path.exists(self.airtable_records):
                raise FileNotFoundError(
                    f"Records file not found: {self.airtable_records}")

            total_records = sum(1 for _ in open(self.airtable_records, 'r'))
            processed_records = 0
            batch_texts = []
            batch_word_count = 0
            batch_char_count = 0
            MAX_BATCH_WORDS = 2000
            MAX_BATCH_CHARS = 8000

            # 添加计数器跟踪不同原因的丢失
            skipped_size = 0
            json_errors = 0
            api_errors = 0

            print(f"开始处理 {total_records} 条记录")

            with open(self.airtable_records, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    try:
                        record = json.loads(line)
                        document_text = str(record)

                        current_words = len(document_text.split())
                        current_chars = len(document_text)

                        if current_words > MAX_BATCH_WORDS or current_chars > MAX_BATCH_CHARS:
                            skipped_size += 1
                            print(
                                f"跳过过大记录 - 词数: {current_words}/{MAX_BATCH_WORDS}, 字符数: {current_chars}/{MAX_BATCH_CHARS}"
                            )
                            continue

                        # 检查添加此记录是否会超出批处理限制
                        if (batch_word_count + current_words >= MAX_BATCH_WORDS
                                or batch_char_count + current_chars
                                >= MAX_BATCH_CHARS):
                            # 如果会超出限制，先处理当前批次
                            if batch_texts:
                                print(f"批次已满，处理当前 {len(batch_texts)} 条记录")
                                try:
                                    print("开始向量化批次...")
                                    if processed_records == 0:
                                        self.vectorstore = FAISS.from_texts(
                                            batch_texts, self.embeddings)
                                    else:
                                        self.vectorstore.add_texts(batch_texts)

                                    processed_records += len(batch_texts)
                                    batch_size = len(batch_texts)
                                    remaining_records = total_records - processed_records
                                    print(
                                        f"批次处理完成: {batch_size} 条记录 | 进度: {processed_records}/{total_records} | 剩余: {remaining_records}"
                                    )

                                    if processed_records >= 100 and processed_records % 100 <= len(
                                            batch_texts):
                                        self.vectorstore.save_local(
                                            self.store_path)

                                except OpenAIError as e:
                                    api_errors += len(batch_texts)
                                    print(f"OpenAI API 错误: {str(e)}")

                                    # 添加 API 错误重试逻辑
                                    try:
                                        # 将批次分成更小的子批次
                                        sub_batch_size = max(
                                            1,
                                            len(batch_texts) // 4)
                                        for i in range(0, len(batch_texts),
                                                       sub_batch_size):
                                            sub_batch = batch_texts[
                                                i:i + sub_batch_size]
                                            print(
                                                f"处理子批次 {i//sub_batch_size + 1}/{(len(batch_texts) + sub_batch_size - 1)//sub_batch_size}，大小: {len(sub_batch)}"
                                            )

                                            # 添加延迟以避免速率限制
                                            if i > 0:
                                                time.sleep(5)

                                            try:
                                                if processed_records == 0 and i == 0:
                                                    self.vectorstore = FAISS.from_texts(
                                                        sub_batch,
                                                        self.embeddings)
                                                else:
                                                    self.vectorstore.add_texts(
                                                        sub_batch)

                                                processed_records += len(
                                                    sub_batch)
                                                print(
                                                    f"子批次处理成功: {len(sub_batch)} 条记录"
                                                )
                                            except Exception as sub_e:
                                                api_errors += len(sub_batch)
                                    except Exception:
                                        print(f"重试逻辑失败: {str(retry_e)}")
                                        pass

                            # 重置批次
                            batch_texts = []
                            batch_word_count = 0
                            batch_char_count = 0

                        # 将当前文档添加到批次
                        batch_texts.append(document_text)
                        batch_word_count += current_words
                        batch_char_count += current_chars

                    except json.JSONDecodeError:
                        json_errors += 1
                        continue
                    except Exception as e:
                        print(f"处理记录 {line_num} 时发生未知错误: {str(e)}")
                        print(f"错误详情: {traceback.format_exc()}")
                        continue

            # 处理最后剩余的文档
            if batch_texts:
                print(f"\n处理最后的批次: {len(batch_texts)} 条记录")
                try:
                    if processed_records == 0:
                        self.vectorstore = FAISS.from_texts(
                            batch_texts, self.embeddings)
                    else:
                        self.vectorstore.add_texts(batch_texts)

                    processed_records += len(batch_texts)
                except OpenAIError as e:
                    api_errors += len(batch_texts)

            # 保存向量存储
            self.vectorstore.save_local(self.store_path)
            print(f"向量存储创建完成，处理了 {processed_records}/{total_records} 条记录")

            # 输出简化的统计信息
            print(
                f"总记录数: {total_records}, 成功处理: {processed_records}, 因大小跳过: {skipped_size}, 因JSON解析错误跳过: {json_errors}, 因API错误跳过: {api_errors}"
            )

            return self.vectorstore

        except Exception as e:
            print(f"处理文档时发生错误: {str(e)}")
            raise

    def _load_system_prompt(self):
        """
        Load system prompt from the configuration file
        
        Returns:
            ChatPromptTemplate|None: The formatted system prompt if successfully loaded, None otherwise
        """
        try:
            if os.path.exists(self.prompt_file):
                with open(self.prompt_file, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
                    system_message = SystemMessagePromptTemplate.from_template(
                        prompts['speaker_recommendation']['prompt'])

                    prompt = ChatPromptTemplate.from_messages([
                        system_message,
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template(
                            "{context}\n\nQuestion: {question}")
                    ])

                print("Successfully loaded system prompt from file")
                return prompt
            else:
                print(
                    f"Prompt file not found at {self.prompt_file}, no prompt will be used"
                )
                return None
        except Exception as e:
            print(
                f"Error loading system prompts: {str(e)}, no prompt will be used"
            )
            raise

    # todo: add prompt rules for all the answers
    def get_answer(self, question, chat_history):
        try:
            # ensure chat_history is a list
            if not isinstance(chat_history, list):
                chat_history = []

            # format the chat history
            formatted_history = []
            if chat_history:  # if chat_history is not empty
                for message in chat_history:
                    if hasattr(message, 'user_message') and hasattr(
                            message, 'ai_response'):
                        formatted_history.append(
                            HumanMessage(content=str(message.user_message)))
                        formatted_history.append(
                            AIMessage(content=str(message.ai_response)))

            # 添加相关文档检索的调试信息
            # retriever = self.vectorstore.as_retriever()
            # relevant_docs = retriever.get_relevant_documents(question)
            # print(f"\nFound {len(relevant_docs)} relevant documents for question: {question}")
            # print("Top 3 most relevant documents:")
            # for i, doc in enumerate(relevant_docs[:3]):
            #     print(f"\nDocument {i+1}:")
            #     print(f"Content: {doc.page_content[:200]}...")
            #     print(f"Similarity score: {doc.metadata.get('score', 'N/A')}")

            #  方法一：切换检索策略，使用MMR代替纯相似度搜索
            # retriever = self.vectorstore.as_retriever(
            #     search_type="mmr",  # 最大边际相关性
            #     search_kwargs={
            #         "k": 8,  # 最终返回的文档数
            #         "fetch_k": 20,  # 初始候选池大小
            #         "lambda_mult": 0.7  # 相关性权重(0.7相关性，0.3多样性)
            #     }
            # )

            # 方法二： 降低相似度阈值
            # retriever = self.vectorstore.as_retriever(
            #     search_type="similarity_score_threshold",
            #     search_kwargs={
            #         "score_threshold": 0.3,  # 从较高值(如0.7)降低到0.3-0.5
            #         "k": 10  # 增加返回文档数量
            #     }
            # )

            # 查询扩展，使用LLM扩展原始查询
            # from langchain.retrievers import ContextualCompressionRetriever
            # from langchain.retrievers.document_compressors import LLMChainExtractor

            # # 创建查询扩展器
            # llm = ChatOpenAI(temperature=0)
            # compressor = LLMChainExtractor.from_llm(llm)

            # # 应用到检索器
            # compression_retriever = ContextualCompressionRetriever(
            #     base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
            #     base_compressor=compressor
            # )

            # 2. 组合关键词搜索和向量搜索

            # 向量检索
            vector_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 10})

            # 获取向量存储中的文档
            start_time = time.time()
            logging.info("开始从 FAISS 向量存储获取文档...")

            # try:
            # raw_documents = self.vectorstore.get_all_documents()
            raw_documents = self.vectorstore.similarity_search("", k=1000)

            # # 检查是否为 FAISS 向量存储
            # if isinstance(self.vectorstore, FAISS):

            #     # 直接从 docstore 获取所有文档
            #     if hasattr(self.vectorstore, "docstore") and hasattr(self.vectorstore.docstore, "docs"):
            #         raw_documents = list(self.vectorstore.docstore.docs.values())
            #         doc_count = len(raw_documents)
            #         elapsed_time = time.time() - start_time
            #         logging.info(f"成功从 FAISS docstore 获取文档: {doc_count} 个文档, 耗时 {elapsed_time:.2f} 秒")
            #     else:
            #         raise AttributeError("FAISS 向量存储结构不符合预期")
            # else:
            #     # 非 FAISS 向量存储，尝试其他方法
            #     logging.info("检测到非 FAISS 向量存储，尝试其他方法获取文档...")
            #     if hasattr(self.vectorstore, "get_all_documents"):
            #         raw_documents = self.vectorstore.get_all_documents()
            #     else:
            #         # 使用 similarity_search
            #         raw_documents = self.vectorstore.similarity_search("", k=1000)

            #     doc_count = len(raw_documents)
            #     elapsed_time = time.time() - start_time
            #     logging.info(f"成功获取文档: {doc_count} 个文档, 耗时 {elapsed_time:.2f} 秒")

            # except Exception as e:
            #     logging.error(f"获取文档失败: {str(e)}")
            #     # 降级方案
            #     logging.info("切换到备用方案: 使用 similarity_search 获取少量文档样本")
            #     raw_documents = self.vectorstore.similarity_search("", k=100)
            #     logging.info(f"获取到 {len(raw_documents)} 个文档样本")

            # 关键词检索
            keyword_retriever = BM25Retriever.from_documents(raw_documents)
            keyword_retriever.k = 10

            # 组合检索器
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, keyword_retriever],
                weights=[0.7, 0.3])

            # # 获取检索结果
            # docs = ensemble_retriever.get_relevant_documents(question)

            # # 后处理：过滤低相似度文档
            # filtered_docs = []
            # for doc in docs:
            #     # 计算与查询的相似度
            #     similarity = self._compute_similarity(question, doc.page_content)
            #     if similarity >= 0.15:  # 应用相似度阈值
            #         filtered_docs.append(doc)

            # 使用过滤后的文档
            self.chain.retriever = ensemble_retriever

            response = self.chain.invoke({
                "question": str(question),
                "chat_history": formatted_history
            })
            return response['answer']
        except Exception as e:
            return f"Sorry, there was an error processing your request: {str(e)}"

    def check_and_clean_vectorstore(self):
        """
        check and clean the existing vector store and record file
        """
        try:
            # if the vector store directory exists, clean up all the files in it
            if os.path.exists(self.store_path):
                print(
                    f"Found existing vector store, cleaning up files in {self.store_path}..."
                )
                for filename in os.listdir(self.store_path):
                    file_path = os.path.join(self.store_path, filename)
                    try:
                        os.remove(file_path)
                        print(f"Removed: {filename}")
                    except Exception as e:
                        print(f"Error removing {filename}: {str(e)}")
                print("Vector store files cleaned up")

        except Exception as e:
            print(f"Error during vector store cleanup: {str(e)}")
            raise

    # def _compute_similarity(self, query, document_text):
    #     # Implement your similarity computation logic here
    #     # This is a placeholder and should be replaced with the actual implementation
    #     return 0.5  # Placeholder similarity score

    # def get_representative_documents(self, vectorstore, sample_size=10000):
    #     """获取代表性文档样本"""
    #     # 方法1：随机采样
    #     if hasattr(vectorstore, "get_random_documents"):
    #         return vectorstore.get_random_documents(sample_size)

    #     # 方法2：使用聚类中心
    #     if hasattr(vectorstore, "get_cluster_centers"):
    #         return vectorstore.get_cluster_centers(num_clusters=min(sample_size, 1000))

    #     # 方法3：使用多样化查询获取不同领域的文档
    #     diverse_docs = []
    #     diverse_queries = ["技术", "管理", "市场", "产品", "金融", "医疗", "教育", "科学", "艺术", "体育"]
    #     docs_per_query = sample_size // len(diverse_queries)

    #     for query in diverse_queries:
    #         docs = vectorstore.similarity_search(query, k=docs_per_query)
    #         diverse_docs.extend(docs)

    #     # 去重
    #     return list({doc.id: doc for doc in diverse_docs}.values())

    # def get_faiss_documents_in_batches(self, batch_size=1000, max_docs=50000):
    #     """分批获取 FAISS 向量存储中的文档"""
    #     from langchain_community.vectorstores import FAISS

    #     if not isinstance(self.vectorstore, FAISS):
    #         raise TypeError("此方法仅适用于 FAISS 向量存储")

    #     all_docs = []
    #     start_time = time.time()
    #     logging.info("开始分批获取 FAISS 文档...")

    #     try:
    #         # 获取所有文档ID
    #         if hasattr(self.vectorstore, "docstore") and hasattr(self.vectorstore.docstore, "docs"):
    #             all_ids = list(self.vectorstore.docstore.docs.keys())
    #             total_docs = len(all_ids)
    #             logging.info(f"FAISS 索引中共有 {total_docs} 个文档")

    #             # 分批处理
    #             for i in range(0, min(total_docs, max_docs), batch_size):
    #                 batch_start = time.time()
    #                 end_idx = min(i + batch_size, total_docs)
    #                 batch_ids = all_ids[i:end_idx]

    #                 # 获取这批文档
    #                 batch_docs = [self.vectorstore.docstore.docs[doc_id] for doc_id in batch_ids]
    #                 all_docs.extend(batch_docs)

    #                 batch_time = time.time() - batch_start
    #                 total_time = time.time() - start_time
    #                 logging.info(
    #                     f"批次 {i//batch_size + 1}/{(min(total_docs, max_docs) + batch_size - 1)//batch_size}: "
    #                     f"获取 {len(batch_docs)} 个文档, 批次耗时 {batch_time:.2f}秒, "
    #                     f"总计 {len(all_docs)} 个文档, 总耗时 {total_time:.2f}秒"
    #                 )

    #                 # 如果已经获取足够多的文档，提前停止
    #                 if len(all_docs) >= max_docs:
    #                     logging.info(f"已达到最大文档数限制 ({max_docs})，停止获取")
    #                     break
    #         else:
    #             raise AttributeError("FAISS 向量存储结构不符合预期")

    #     except Exception as e:
    #         logging.error(f"分批获取 FAISS 文档时出错: {str(e)}")
    #         if len(all_docs) == 0:
    #             # 如果完全失败，使用备用方案
    #             logging.info("使用备用方案获取少量文档样本")
    #             all_docs = self.vectorstore.similarity_search("", k=100)

    #     logging.info(f"FAISS 文档获取完成，共 {len(all_docs)} 个文档，总耗时 {time.time() - start_time:.2f}秒")
    #     return all_docs

    # # 使用分批方法获取文档
    # try:
    #     raw_documents = self.get_faiss_documents_in_batches(batch_size=1000, max_docs=10000)
    # except Exception as e:
    #     logging.error(f"分批获取文档失败: {str(e)}")
    #     # 降级方案
    #     raw_documents = self.vectorstore.similarity_search("", k=100)
    #     logging.info(f"降级方案: 获取到 {len(raw_documents)} 个文档样本")
