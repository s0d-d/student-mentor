from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

persist_directory="./chroma_db"
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


llm=LlamaCpp(
    model_path="models/llama-2-7b-chat.Q5_K_S.gguf.bin",
    temperature=0.1,
    max_tokens=2000,
    top_p=1,
    n_ctx=4096,  # Context window
    #n_batch=512,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager,
    stop = ['### Human:', '### Assistant:']
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

vector_retriever = vectordb.as_retriever(
    #search_type="similarity_score_threshold", 
    #search_kwargs={"score_threshold": 0.5}
    search_kwargs={"k": 1}
)

template = """
Assistant is a large language model.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
Answer the question from Human based only on the given context. 
If you do not have information in the context, say you do not know, instead of making up on your own.

Context: {context}

### Human: {question}

### Assistant answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {"context": vector_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("ELTE student mentor: How can I help you today?")
while True:
    query = input("You: ")
    docs = vector_retriever.invoke(query)
    print(docs)

    chain.invoke(query)





