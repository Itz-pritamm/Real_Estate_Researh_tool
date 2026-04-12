from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import WebBaseLoader
from uuid import uuid4
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
# Chains

from langchain_community.document_loaders import UnstructuredURLLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector DB (Chroma)
from langchain_community.vectorstores import Chroma

# LLM (Groq)
from langchain_groq import ChatGroq

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

prompt=PromptTemplate.from_template(
    """Answer the question based only on the provided context.

Context:
{context}

Question:
{input}

Answer:"""
)
load_dotenv()

#constant
CHUNK_SIZE=1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"
llm=None
vector_store=None
def initialize_components():
    global llm , vector_store
    if llm is None:
    
    
        llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0.9,max_tokens=500)
    if vector_store is None:
        ef=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code":True}
        )
        vector_store=Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)

        )
def process_urls(urls):
    """
    This function scraps the data from a urls and store it in vector db
    :param urls:input urls
    :return:
    """
    yield "initiaze components"
    initialize_components()
    

    #vector_store.delete_collection()
    yield "load data"
    loader = WebBaseLoader(urls)
    data=loader.load()

    yield "spliiiting the text"
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_size=CHUNK_SIZE

    )

    docs=text_splitter.split_documents(data)

    yield "adding docs to vector db"
    uuids=[str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=uuids)

    yield "Done adding docs to vector database..."

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("vector database is not initialized")
    retriever = vector_store.as_retriever()

    document_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    # Run query
    result = chain.invoke({"input": query})

    # Extract answer
    answer = result.get("answer", "")

    # Extract sources (as string like old version)
    sources = ", ".join(
        [str(doc.metadata.get("source", "")) for doc in result.get("context", [])]
    )

    return answer, sources
    



if __name__=="__main__":
    urls=[
        "https://www.cnbc.com/2026/04/08/fed-officials-still-foresee-rate-cut-this-year-despite-war-impacts-minutes-show.html",
        "https://www.cnbc.com/mortgages/"
    ]
    process_urls(urls)
    answer,sources=generate_answer("what was Gross domestic product rose at just _ in fourth quarter of 2025?")
    print(f"anwer:{answer}")
    print(f"sources:{sources}")