import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Path to FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

# Streamlit UI Setup
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

# Sidebar settings
with st.sidebar:
    st.header("Chatbot Settings ⚙️")
    HUGGINGFACE_REPO_ID = st.selectbox(
        "Select Model",
        ["mistralai/Mistral-7B-Instruct-v0.3", "HuggingFaceH4/zephyr-7b-beta"],
        index=0
    )
    response_temp = st.slider("Response Temperature", 0.0, 1.0, 0.5, 0.1)
    st.markdown("---")
    st.subheader("About")
    st.markdown("🤖 **AI Chatbot** powered by **LangChain** & **HuggingFace**.")


@st.cache_resource
def get_vectorstore():
    """Load FAISS vector store with sentence-transformers embeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return vectorstore


def set_custom_prompt():
    """Set custom instruction prompt for chatbot."""
    return PromptTemplate(

        template="""
        You are an advanced AI assistant specializing in **medical diagnosis and healthcare insights**.  
        Use the provided **context** to deliver accurate, relevant, and professional responses.  

        ---

        ### 🔹 **Response Guidelines:**  
        ✅ **If the context contains medical information**, provide a **clear and concise diagnosis or advice.**  
        ❌ **If the context lacks relevant medical data**, respond with: **"I don't have enough information to provide a diagnosis."**  
        🔍 **If the question is unclear**, ask for **additional symptoms, patient history, or clarification.**  
        📌 **For step-by-step medical guidance**, present the information logically and in an easy-to-understand format.  
        🚨 **If the query requires urgent medical attention**, advise:  
           _"Please consult a healthcare professional immediately."_  

        ---

        ### **Context (Medical Data / Symptoms / Reports):**  
        {context}  

        ### **User Query (Health Concern / Diagnosis Request):**  
        {question}  

        💡 **AI Medical Response:** (Start directly)
        """,
        input_variables = ["context", "question"]
    )


def load_llm(repo_id, HF_TOKEN, temperature):
    """Load HuggingFace LLM with provided repo ID and token."""
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature=temperature,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )


def main():
    """Streamlit Chatbot UI and Logic"""
    st.title("🤖 AI Chatbot - Professional Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Ask me anything...")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        HF_TOKEN = os.environ.get("HF_TOKEN")
        vectorstore = get_vectorstore()

        if vectorstore is None:
            st.error("Error: Could not load the vector store.")
            return

        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN, response_temp),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})
            answer = response.get("result", "I'm sorry, I couldn't process that.")
            sources = response.get("source_documents", [])

            # Display chatbot response
            with st.chat_message("assistant"):
                st.markdown(answer)

                if sources:
                    with st.expander("📌 Source Documents"):
                        for doc in sources:
                            st.markdown(f"- **{doc.metadata.get('source', 'Unknown')}**: {doc.page_content[:300]}...")

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
