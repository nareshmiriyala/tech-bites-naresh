# pip install faiss-cpu
# pip install sentence-transformers
#https://huggingface.co/sentence-transformers
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# import
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, UnstructuredHTMLLoader

from demos.queries import QUERY2_OPTIMA_FUNDED

FAISS_DB = "faiss_db"

# load the document and split it into chunks
loader = UnstructuredHTMLLoader("PEXA continues UK expansion with Optima Legal acquisition - PEXA.html")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
# We can also use OpenAIEmbedding
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embedding_function)
# save to disk
db.save_local(FAISS_DB)
# load from disk
db2 = FAISS.load_local(FAISS_DB, embedding_function)
# query it
# query = QUERY1_OPTIMA_ACQUIRED
query = QUERY2_OPTIMA_FUNDED
docs = db2.similarity_search(query)

# print results
print(docs[0].page_content)
