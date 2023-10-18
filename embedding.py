from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings

load_dotenv()
embedding_function = OpenAIEmbeddings()

text = "This is a demo document."

query_result = embedding_function.embed_query(text)

print("OpenAI Embedding:", query_result[:7])

sentence_embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

query_result_os = sentence_embedding_function.embed_query(text)

print("Open Source Embedding:", query_result_os[:7])
