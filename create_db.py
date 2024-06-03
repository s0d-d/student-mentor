from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load and process the text
#loader = TextLoader('elte_history.txt')
# loader = DirectoryLoader('./data', glob="**/[!.]*.txt", show_progress=True)


import os
import fnmatch

def find_text_files(directory, pattern='*.txt'):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

directory_to_search = './data-new'  # directory to load recursively
text_files = find_text_files(directory_to_search)

loaders = []
for file in text_files:
    loaders.append(TextLoader(file))


documents = []
for loader in loaders:
    documents.extend(loader.load())


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk
persist_directory = './chroma_db'

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2') 
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

vectordb.persist()
