import streamlit as st
import os
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from qdrant_client import QdrantClient

# Initialize Qdrant client
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def init_vectorstore(collection_name: str) -> Qdrant:
    """Initialize Qdrant vectorstore."""
    return Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=OpenAIEmbeddings(),
    )


def process_urls(urls: List[str], collection_name: str):
    """Process URLs and add them to Qdrant."""
    # Load documents
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading URL {url}: {str(e)}")
            continue

    if not docs:
        return False, "No documents were loaded successfully."

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents(docs)

    # Add to vectorstore
    vectorstore = Qdrant.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        collection_name=collection_name,
    )

    return True, f"Successfully processed {len(doc_splits)} document chunks."


def get_collections():
    """Get list of all collections."""
    try:
        collections = qdrant_client.get_collections()
        return [collection.name for collection in collections.collections]
    except Exception as e:
        return {"error": str(e)}


def chat_with_docs(query: str, collection_name: str) -> str:
    """Chat with the documents using RAG."""
    # Initialize vectorstore
    vectorstore = init_vectorstore(collection_name)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Get relevant documents
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "No relevant documents found to answer your question."

    # Setup RAG prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Setup LLM
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.7,
        streaming=True
    )

    # Create and run chain
    rag_chain = prompt | llm | StrOutputParser()

    # Format documents
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate response
    response = rag_chain.invoke({
        "context": context,
        "question": query
    })

    return response


# Streamlit UI
st.title("Document Management and Chat System")

# Initialize session state for OpenAI API key
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Sidebar for API key
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.openai_api_key = api_key

if not st.session_state.openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    st.stop()

# Main tabs
tab1, tab2 = st.tabs(["Add Documents", "Chat"])

with tab1:
    st.subheader("Add Documents from URLs")

    # Collection name input
    collection_name = st.text_input("Collection Name", key="add_collection_name")

    # URL inputs
    urls = []
    url_count = st.number_input("Number of URLs", min_value=1, max_value=10, value=1)

    for i in range(url_count):
        url = st.text_input(f"URL {i + 1}")
        if url:
            urls.append(url)

    if st.button("Process URLs"):
        if not collection_name:
            st.error("Please enter a collection name.")
        elif not urls:
            st.error("Please enter at least one URL.")
        else:
            with st.spinner("Processing URLs..."):
                success, message = process_urls(urls, collection_name)
                if success:
                    st.success(message)
                else:
                    st.error(message)

with tab2:
    st.subheader("Chat with Documents")

    # Get collections
    collections = get_collections()
    if isinstance(collections, list):
        if not collections:
            st.warning("No collections available. Please add documents first.")
        else:
            # Collection selection
            selected_collection = st.selectbox(
                "Select Collection",
                options=collections,
                key="chat_collection"
            )

            # Chat interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask a question about your documents"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chat_with_docs(prompt, selected_collection)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error(f"Error fetching collections: {collections['error']}")

# Debug information
with st.expander("Debug Information"):
    st.write("Collections:", get_collections())
    st.write("Environment Variables:", {
        "QDRANT_HOST": QDRANT_HOST,
        "QDRANT_PORT": QDRANT_PORT
    })