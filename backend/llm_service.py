from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os

class LLMService:
    def __init__(self, documents):
        # 设置向量存储的保存路径
        self.store_path = "vector_store"
        
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.embeddings = OpenAIEmbeddings()
        
        try:
            # 确保存储目录存在
            os.makedirs(self.store_path, exist_ok=True)
            
            # 尝试加载现有的向量存储
            if os.path.exists(os.path.join(self.store_path, "index.faiss")):
                print("Loading existing vector store...")
                self.vectorstore = FAISS.load_local(
                    self.store_path,
                    self.embeddings
                )
            else:
                print("Creating new vector store...")
                # 确保文档格式正确
                if not documents:
                    documents = ["这是一个初始化文档"]
                else:
                    documents = [str(doc) if not isinstance(doc, str) else doc for doc in documents]
                
                # 创建新的向量存储
                self.vectorstore = FAISS.from_texts(documents, self.embeddings)
                # 保存向量存储到本地
                print("Saving new vector store...")
                self.vectorstore.save_local(self.store_path)
            
            self.llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo"

            
            self.chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=self.vectorstore.as_retriever(),
                return_source_documents=True
            )
            
        except Exception as e:
            print(f"初始化错误: {str(e)}")
            raise

    def add_documents(self, new_documents):
        """添加新文档到向量存储"""
        try:
            self.vectorstore.add_texts(new_documents)
            # 保存更新后的向量存储
            self.vectorstore.save_local(self.store_path)
            return True
        except Exception as e:
            print(f"添加文档错误: {str(e)}")
            return False

    # todo: add prompt rules for all the answers
    def get_answer(self, question, chat_history):
        try:
            formatted_history = [(q, a) for q, a in chat_history]
            response = self.chain({"question": question, "chat_history": formatted_history})
            return response['answer']
        except Exception as e:
            print(f"Error in get_answer: {str(e)}")
            return f"抱歉，处理您的问题时出现错误: {str(e)}"