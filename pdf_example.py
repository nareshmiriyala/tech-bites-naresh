# pip install pypdf
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

GPT_3 = "gpt-3.5-turbo"
GPT_3_16K = "gpt-3.5-turbo-16k"
GPT_4 = "gpt-4"
gpt4all_llm_model = "../models/llama-2-7b-chat.ggmlv3.q4_0.bin"


def get_address_from_house_contract(model, pdf_file, query_llm):
    # Conveyancers Create workspace from signed contract pdf,
    # I am using section 32 as I don't have signed contract pdf as example
    pdf_faiss_index = "pdf_faiss_index_signed_contract"+pdf_file
    loader = PyPDFLoader(pdf_file)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    pages = loader.load_and_split()
    load_dotenv()

    model_kwargs = {
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    embedding_function = OpenAIEmbeddings()
    db = FAISS.from_documents(pages, embedding_function)
    db.save_local(pdf_faiss_index)

    # load from disk
    db2 = FAISS.load_local(pdf_faiss_index, embedding_function)

    def add_prompt():
        RESPONSE_TEMPLATE = """\
       You are an expert Pdf Reader, tasked with answering any question \
       about only Housing Contract Pdfs in Australia.
   
       Generate a comprehensive and informative answer of 80 words or less for the \
       given question based solely on the provided PDF. You must \
       only use information from the provided PDF. Use an unbiased and \
       journalistic tone. Combine search results together into a coherent answer. Do not \
       repeat text.If \
       different results refer to different entities within the same name, write separate \
       answers for each entity.
   
       You should use bullet points in your answer for readability. Put citations where they apply
       rather than putting them all at the end.
   
       If there is nothing in the context relevant to the question at hand, just say "Hmm, \
       I'm not sure." Don't try to make up an answer.
   
       Anything between the following `context`  html blocks is retrieved from a knowledge \
       bank, not part of the conversation with the user. 
   
       <context>
           {context} 
       <context/>
   
       REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
       not sure." Don't try to make up an answer. Anything between the preceding 'context' \
       html blocks is retrieved from a knowledge bank, not part of the conversation with the \
       user.\
       """
        messages = [
            SystemMessagePromptTemplate.from_template(RESPONSE_TEMPLATE),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        return {"prompt": prompt}

    def ask_chatbot(faiss_db, gpt_model):
        if gpt_model == gpt4all_llm_model:
            llm = GPT4All(model=gpt_model)
        else:
            llm = ChatOpenAI(model_name=gpt_model,
                             streaming=True,
                             temperature=0,
                             model_kwargs=model_kwargs
                             )
        chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                      retriever=faiss_db.as_retriever())
        history = []

        query_openai = query_llm
        response = chain({"question": query_openai, "chat_history": history})["answer"]
        return response

    print(ask_chatbot(db2, model))
    print("----------------------")


query_valid_contract = " Is this a valid singed contract, answer yes/no?"
query_house_address = (" What's the house address,vendor full name "
                       "and buyer full name in a single python list between <code></code> block,"
                       " return '' if not found?")

for query in [query_valid_contract, query_house_address]:
    get_address_from_house_contract(GPT_4, "section_32.pdf", query)
    get_address_from_house_contract(GPT_4, "Blank-section-32.pdf", query)
