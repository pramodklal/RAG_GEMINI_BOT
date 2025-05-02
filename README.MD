MultiFiles RAG - ChatBot: Retrieval Augmented Generation (RAG) chatbot using Google's Gemini 2.0 & l.5 model, Langchain, FAISS, and Streamlit
This RAG-ChatBot is a Python application that allows the user to chat with MultiFiles (PDF,DOCX,TXT) documents. You ask questions in natural language, in the same way as if you were to ask a human, and the application will provide relevant responses based on the content of the uploaded documents. This app uses Google's Gemini 2.0 & l.5 model to generate accurate answers to your questions, but the model will only answer questions that are about the uploaded documents. Here are some key points about the project:

Upload Documents: When the app is launched, you can upload a MultiFiles (PDF,DOCX,TXT) documentst and chat with the document on the fly, no need to reload the app

The user interface was crafted with streamlit, with the goal of displaying all necessary information while being extremely simple. The user only has the "upload" button, all the rest is automated by the app
The model incorporates the chat history, retaining up to 10 users questions and model responses, so if you ask about something and want more details, you can just say "give me more  formation about that" and the model will know what you are reffering to
For each response, you can check the source in the sidebar, making sure that the model is not making up responses
