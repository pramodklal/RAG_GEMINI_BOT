import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from docx import Document
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model_choice = {
   # "id1": "Gemini-2.5",
    "id2": "Gemini-2.0",
    "id3": "Gemini-1.5",
}
other_files = []

# read all pdf files and return text

# Extract text from a PDF file

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from a DOCX file

def get_word_text(docx_file):
    document = Document(docx_file)
    text = "\n".join([paragraph.text for paragraph in document.paragraphs])
    return text

# Extract text from a TXT file
def read_text_file(txt_file):
    text = txt_file.getvalue().decode('utf-8')
    return text

# Combine text from different files

def combine_text(text_list):
    return "\n".join(text_list)

# split text into chunks

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    #return vector_store

def get_conversational_chain(modelname):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(#model="gemini-pro",
                                   model=modelname,
                                   #model=selected_model,
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]


def user_input(user_question,modelname):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    #new_db = FAISS.load_local("faiss_index", embeddings) 
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(modelname)
    
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="Multi-Files Chatbot using Gemini",
        #page_icon="🤖"
        page_icon=":books:"
    )
    
    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")

        selected_model=""
        model_list = st.sidebar.selectbox("Choose Model:", list(model_choice.items()), 0 , format_func=lambda o: o[1])
        if (model_list[1]=="Gemini-2.0"):
            selected_model= "gemini-2.0-flash"
        if (model_list[1]=="Gemini-1.5"):
            selected_model = "gemini-1.5-flash-8b"
        files = st.file_uploader(
            "Upload your PDF,DOCX,TXT Files and Click on the Submit & Process Button", accept_multiple_files=True)
        for file1 in files:
             other_files.append(file1)
        pdf_texts = []
        word_texts = []
        txt_texts = []
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                #raw_text = get_pdf_text(pdf_docs)
               for file in other_files:
                        if file.name.lower().endswith('.pdf'):
                            pdf_texts.append(get_pdf_text(file))
                        elif file.name.lower().endswith('.docx'):
                            word_texts.append(get_word_text(file))
                        elif file.name.lower().endswith('.txt'):
                            txt_texts.append(read_text_file(file))
            # Combine text from different file types
            combined_text = combine_text(pdf_texts + word_texts + txt_texts)
            #st.write(combined_text)
            text_chunks = get_text_chunks(combined_text)
            get_vector_store(text_chunks)
            st.success("Done")


    st.image("multifiles.jpg")
    st.markdown("<h2 style='color:purple;vertical-align:top;'>Multi-Files Chatbot using Gemini </h2>", unsafe_allow_html=True)
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
   
    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some pdf,docx,txt and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                #response = user_input(prompt)
                response = user_input(prompt,selected_model)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)
if __name__ == "__main__":
     try:
        main()
     except Exception as e:
        # Manage Errors: Other Exceptions
        st.error(f"An error occurred: {e}")   