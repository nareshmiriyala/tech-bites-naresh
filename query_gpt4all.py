# pip install faiss-cpu
# pip install gpt4all
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import GPT4All
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, UnstructuredHTMLLoader

from demos.queries import QUERY2_OPTIMA_FUNDED

load_dotenv()
FAISS_DB_OPEN_AI = "faiss_db_open_ai"
PEXABOT_FAISS_DB_OPEN_AI = "./../faiss_index"
GPT_3 = "gpt-3.5-turbo"
GPT_3_16K = "gpt-3.5-turbo-16k"
GPT_4 = "gpt-4"
model_kwargs = {
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}
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
# query = QUERY1_OPTIMA_ACQUIRED
query = QUERY2_OPTIMA_FUNDED
docs = db2.similarity_search(query)

# print results
print(f"Similarity_search match:\n {docs[0].page_content}")


def ask_chatbot(faiss_db, gpt_model):
    llm = GPT4All(model=gpt_model, n_predict=512)
    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                  retriever=faiss_db.as_retriever())
    history = []

    query_openai = query
    response = chain({"question": query_openai, "chat_history": history})["answer"]
    return response


glt4all_llm_model = "../models/llama-2-7b-chat.ggmlv3.q4_0.bin"
print(
    f"Ingestion index query response GPT4ALL Model : '{ask_chatbot(FAISS.load_local(FAISS_DB_OPEN_AI, embedding_function), glt4all_llm_model)}' ")
