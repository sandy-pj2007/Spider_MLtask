import streamlit as st
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

st.set_page_config(page_title="Research Paper RAG")

st.title("📚 Research Paper RAG Assistant")
api_key = st.text_input(
    "Spider_task_api",
    type="password"
)
@st.cache_resource
def build_retriever():

    pdf_files = [
        "Sentence-BERT.pdf",
        "GPT.pdf",
        "RAG-NLP.pdf",
        "Llama2.pdf",
        "BERT.pdf",
        "LoRA.pdf",
        "attentionisallyouneed.pdf"
    ]

    documents = []

    for pdf in pdf_files:

        loader = PyPDFLoader(pdf)

        documents.extend(
            loader.load()
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding_model,
        persist_directory="./chroma_db"
    )

    return vectordb.as_retriever(
        search_kwargs={"k":3}
    )


with st.spinner("Building knowledge base... First run may take a few minutes."):

    retriever = build_retriever()

st.success("Knowledge base ready!")
if api_key:   
   client = genai.Client(api_key=api_key)

query = st.text_input("Ask a question")

if st.button("Submit"):
    if not api_key:

       st.error("Enter Gemini API key")

       st.stop()

    if not query:

       st.error("Enter a question")

       st.stop()
    docs = retriever.invoke(query)
    

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt =f"""
Context:
{context}

Question:
{query}

Using ONLY the information in the context, answer the question.

If the context is COMPLETELY unrelated to the question, say:
'Insufficient information.'

Otherwise answer normally.

also mention sources
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    st.subheader("Answer")
    st.write(response.text)

    st.subheader("Sources")

    shown = set()

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "Unknown Source"
        )

        if source not in shown:

            st.write(source)
            shown.add(source)