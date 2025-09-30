from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI


class ChatBot:
    load_dotenv()
    def __init__(self):
        base = os.path.dirname(__file__)  # directory where RAG_ChatBot.py lives
        txt_path = os.path.join(base, "materials", "torontoTravelAssistant.txt")

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Expected data file not found: {txt_path}")

        # Load and split documents
        loader = TextLoader(txt_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings()

        # Initialize Pinecone instance
        pc = Pinecone(api_key= os.getenv('PINECONE_API_KEY'))

        index_name = "langchain-demo"

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )            
            )
        index = pc.Index(index_name)
        docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)

        # Initialize ChatOpenAI
        llm = GoogleGenerativeAI(model="gemini-2.5-flash", api_key=os.getenv('GOOGLE_API_KEY'))
        # model_name = "gpt-3.5-turbo"
        # llm = ChatOpenAI(model_name=model_name, organization='')


        # Define prompt template
        template = """
        You are a Toronto travel assistant. Users will ask you questions about their trip to Toronto. Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know.
        Your answer should be short and concise, no longer than 2 sentences.

        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        self.rag_chain = RetrievalQA.from_chain_type(
            llm, retriever=docsearch.as_retriever(), chain_type_kwargs={"prompt": prompt}
        )

        
# Usage example:
if __name__ == "__main__":
    chatbot = ChatBot()
