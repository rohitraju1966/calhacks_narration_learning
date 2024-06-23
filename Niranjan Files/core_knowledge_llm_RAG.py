import os
import pickle
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
#from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter 
load_dotenv()

def model_embeddings_vectordatabase(pdf,key="OPENAI_API_KEY"):
    api_key = os.getenv(key)
    Model = "gpt-3.5-turbo"   #mistral or GPT with api_key
    #Model = "llama2" #Open-source model
    
    #declaring the model and embeddings
    if Model.startswith("gpt"):
        model = ChatOpenAI(api_key=api_key,model=Model)
        embeddings=OpenAIEmbeddings()
    else:
        model = Ollama(base_url='http://host.docker.internal:11434',model='llama2')
        embeddings=OllamaEmbeddings(base_url='http://host.docker.internal:11434',model="llama2")
    
    #define pages
    loader = PyPDFLoader(pdf)
    pages = loader.load_and_split()

    #vectorstore
    vectorstore=DocArrayInMemorySearch.from_documents(pages,embedding=embeddings)
    
    return vectorstore, model

 
def RAG_inference(question, vectorstore, model,key="OPENAI_API_KEY"):
    
    #parser declaration
    parser=StrOutputParser()
    
    #defining a template for prompt
    template = """
    Answer the question in detial based on the context given below. The answer must cater to elementary school students. If you cannot find the answer from the context or if 
    the context does not exist, Reply "NULL". 

    Context: {context}

    Question: {question}
    """
    #prompt declaration
    prompt=PromptTemplate.from_template(template)
    retriever = vectorstore.as_retriever()
    
    #Model pipeline
    chain = (
    {
        "context": itemgetter("question")|retriever,
        "question": itemgetter("question")
    }
    |prompt
    |model
    |parser
    )
    Questions = [
    question
    ]
    for question in Questions:
        #print(f"Question:{question}")
        answer= chain.invoke({'question':question})
        #print(f"Answer: {answer}")
        
    return answer