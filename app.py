import os
import time
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document

# Function to embed vectors from a DataFrame and save them to FAISS vector store
def vector_embedding(df):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create documents with all columns concatenated for page content and save metadata
    docs = [
        Document(
            page_content=" ".join([f"{col}: {row[col]}" for col in df.columns]),
            metadata={col: row[col] for col in df.columns}
        ) for _, row in df.iterrows()
    ]
    print(f"Created {len(docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_documents = text_splitter.split_documents(docs)
    print(f"Split into {len(final_documents)} final documents")

    # Extract text content from final documents
    texts = [doc.page_content for doc in final_documents]

    # Embed the texts
    embeddings_matrix = embeddings.embed_documents(texts)
    print(f"Created embeddings for {len(embeddings_matrix)} documents")

    # Create Document objects with embeddings
    embedded_docs = [
        Document(page_content=doc.page_content, metadata=doc.metadata, embedding=embedding)
        for doc, embedding in zip(final_documents, embeddings_matrix)
    ]

    # Save the embedded documents to FAISS vector store
    vectors = FAISS.from_documents(embedded_docs, embeddings)
    print(f"FAISS vector store created with {vectors.index.ntotal} vectors")

    return vectors

def main():
    # Set dataset path
    dataset_path = "/home/umar/Documents/DATA_MINING/L1/movies.csv"

    # Load dataset using CSVLoader from LangChain
    loader = CSVLoader(file_path=dataset_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    df = pd.DataFrame(data)
    
    # Convert dataset to FAISS vector database
    print("Converting dataset to vector database...")
    vectors = vector_embedding(df)
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    vectors.save_local(DB_FAISS_PATH)
    
    # Initialize ChatGroq with FAISS vectors
    groq_api_key = "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"
    chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    
    # Prompt template for ChatGroq
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
    
    # Create retrieval chain using LangChain
    retriever = vectors.as_retriever()
    document_chain = create_stuff_documents_chain(chat_groq, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # User input and response handling
    while True:
        user_prompt = input("Input your question here: ")
        if user_prompt.lower() in ['exit', 'quit']:
            print("Exiting the application.")
            break
        
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        response_time = time.process_time() - start
        
        # Display response and related documents
        print(f"Response time: {response_time:.2f} seconds")
        print(f"Response: {response['answer']}")
        
        # print("\nDocument Similarity Search:")
        # for i, doc in enumerate(response["context"]):
        #     print(f"\nDocument {i + 1}:")
        #     print(doc.page_content)
        #     print("--------------------------------")

if __name__ == "__main__":
    main()
