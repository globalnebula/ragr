import PyPDF2
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Step 2: Chunk text into manageable sections
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 3: Retrieve relevant context based on user query
def retrieve_context(query, chunks):
    # Create TF-IDF vectors for the chunks and query
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks + [query])
    query_vector = vectors[-1]
    chunk_vectors = vectors[:-1]
    
    # Calculate cosine similarity between query and chunks
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]  # Retrieve top 3 similar chunks
    
    # Fetch top chunks as context
    top_chunks = [chunks[i] for i in top_indices]
    return top_chunks

# Step 4: Fetch answer using Groq AI
def fetch_rag_answer(user_query, pdf_text, client):
    # Process PDF content into chunks
    chunks = chunk_text(pdf_text)
    
    # Retrieve relevant context
    retrieved_chunks = retrieve_context(user_query, chunks)
    context = "\n\n".join(retrieved_chunks)
    
    # Interact with Groq AI using the correct API usage format
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a knowledgeable assistant providing context-based answers."
            },
            {
                "role": "assistant",
                "content": f"Context for query:\n{context}"
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    # Stream the response
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    return response

# Step 5: Streamlit Application
def main():
    st.title("PDF Question Answering System")
    st.subheader("Ask your question based on the fixed PDF")

    # User input query
    user_query = st.text_input("Enter your question:")
    
    if user_query:
        st.write("Processing your query...")
        
        # Path to the fixed PDF file
        pdf_path = "example.pdf"
        
        # Extract PDF text
        pdf_text = extract_text_from_pdf(pdf_path)
        
        groq_api_key = "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"
        client = Groq(api_key=groq_api_key)
        
        # Fetch RAG answer
        try:
            answer = fetch_rag_answer(user_query, pdf_text, client)
            st.subheader("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
