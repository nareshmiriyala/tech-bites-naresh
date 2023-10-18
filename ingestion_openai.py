# pip install faiss-cpu
# pip install openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, UnstructuredHTMLLoader

from demos.queries import QUERY1_OPTIMA_ACQUIRED, QUERY2_OPTIMA_FUNDED

load_dotenv()
FAISS_DB_OPEN_AI = "faiss_db_open_ai"

# load the document and split it into chunks
loader = UnstructuredHTMLLoader("PEXA continues UK expansion with Optima Legal acquisition - PEXA.html")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embedding_function = OpenAIEmbeddings()

db = FAISS.from_documents(docs, embedding_function)
# save to disk
db.save_local(FAISS_DB_OPEN_AI)
# load from disk
db2 = FAISS.load_local(FAISS_DB_OPEN_AI, embedding_function)
# query it
# query = QUERY1_OPTIMA_ACQUIRED
query = QUERY2_OPTIMA_FUNDED
docs = db2.similarity_search(query)

# print results
print(docs[0].page_content)
