from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain


def get_documents_from_pdf():
    file_path = 'data\Gragas_Biography.pdf'
    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()   
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 200
    )
    
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embeddings = OllamaEmbeddings(
        model = "llama2"
    )
    vectorStore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorStore

def create_chain(vectorStore):
    
    model = ChatOllama(
        model = 'llama2',
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {input}
    """)
    
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    
    retriever = vectorStore.as_retriever()
    
    retrieval_chain = create_retrieval_chain(
        retriever,
        chain
    )
    
    return retrieval_chain

    
docs = get_documents_from_pdf()
vectorStore = create_db(docs)
chain = create_chain(vectorStore)

response = chain.invoke({
        "input": "What Gragas love?"
    })

print(response["answer"])
