from flask import Flask, request
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
    #search_kwargs={"score_threshold": 0.1}
    search_kwargs={"k": 3}
    #search_type="mmr"
)

template = """
<s>[INST]<<SYS>>
Assistant is a large language model from ELTE university. Answer the question from ### Human based only on the GIVEN context BELOW. Answer in a short and consice way. Assistant is designed to be able to assist students providing information about ELTE university. If you do not have information in the context, say you do not know.

Context: {context}
<</SYS>>
### Human: {question}
[/INST]
### Assistant answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n".join([d.page_content for d in docs])

chain = (
    {"context": vector_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#print("ELTE student mentor: How can I help you today?")
#while True:
#    query = input("You: ")
#    docs = vector_retriever.invoke(query)
#    print(docs)
#
#    chain.invoke(query)

app = Flask(__name__)

@app.route('/api/endpoint', methods=['POST'])
def call_function():
    # print("hello")
    message = request.json['message']
    docs = vector_retriever.invoke(message)
    print("\n\n_________________\ndocs\n", docs, "\n\n")
    result = chain.invoke(message)
    return result

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(
        host="0.0.0.0", 
        port=5000, 
        ssl_context=(
            './cert/fullchain1.pem', 
            './cert/privkey1.pem'
        )
    )


