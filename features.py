import os
import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import tempfile
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage
from groq import Groq
from elevenlabs.client import ElevenLabs
from elevenlabs import save
from pydub import AudioSegment
from pydantic import BaseModel, Field
import base64
import shutil

# Define QA format for podcast content
class QAFormat(BaseModel):
    question: str = Field(..., description="Generated question")
    answer: str = Field(..., description="Generated answer")

# App configuration with fixed API keys
class Config:
    MISTRAL_API_KEY = ""
    GROQ_API_KEY = ""
    ELEVENLABS_API_KEY = ""

# Initialize clients
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatMistralAI(
        model="mistral-large-latest", 
        temperature=0,
        api_key=Config.MISTRAL_API_KEY
    )

def init_groq_client():
    return Groq(api_key=Config.GROQ_API_KEY)

def init_elevenlabs_client():
    return ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)

# Initialize session state variables
def init_session_state():
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    if 'chunked_texts' not in st.session_state:
        st.session_state.chunked_texts = None
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'podcast_file' not in st.session_state:
        st.session_state.podcast_file = None
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = None

# PDF processing functions
def extract_text_from_pdf(pdf_file):
    """Extracts text from a given PDF file with proper cleanup."""
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory
    tmp_path = os.path.join(temp_dir, "temp_pdf.pdf")
    
    try:
        # Save the uploaded file to the temporary directory
        with open(tmp_path, 'wb') as f:
            f.write(pdf_file.getvalue())
        
        # Open and process the PDF
        doc = fitz.open(tmp_path)
        text = "".join(page.get_text("text") + "\n\n" for page in doc).strip()
        doc.close()  # Explicitly close the document
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None
    finally:
        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            st.warning(f"Failed to clean up temporary files: {e}")

def chunk_text(text, chunk_size=2000, chunk_overlap=200):
    """Splits text into smaller chunks for embedding retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

def create_faiss_index(chunks, embedding_model):
    """Creates a FAISS index from the research paper chunks."""
    try:
        with st.spinner("Creating embeddings from text chunks..."):
            embeddings = embedding_model.encode(chunks, show_progress_bar=False)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings, dtype=np.float32))
            return index, embeddings
    except Exception as e:
        st.error(f"Error creating FAISS index: {e}")
        return None, None

def search_faiss(query, k=3):
    """Finds the most relevant chunks using FAISS."""
    if not st.session_state.faiss_index:
        return []
    
    embedding_model = load_embedding_model()
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = st.session_state.faiss_index.search(query_embedding, k)
    return [st.session_state.chunked_texts[i] for i in indices[0] if i < len(st.session_state.chunked_texts)]

# AI Processing functions
def summarize_retrieved_chunks(retrieved_chunks, llm):
    """Summarizes retrieved chunks using Mistral."""
    if not retrieved_chunks:
        return "No relevant sections found in the document."

    prompt = """
    ### System Role:
    You are a highly skilled AI research assistant with expertise in summarizing complex research papers. Your goal is to extract and condense the most critical information from the provided text into a concise and coherent summary.

    ### Instructions:
    1. **Focus on Key Sections**: Summarize the following sections of the research paper:
       - **Introduction**: Clearly state the purpose of the research and the problem it addresses.
       - **Methodology**: Describe the techniques, tools, and approaches used in the study.
       - **Results**: Highlight the key findings and outcomes of the research.
       - **Conclusion**: Summarize the main takeaways, implications, and any suggested future directions.

    2. **Be Concise**: Avoid unnecessary details. Focus on the most important points.
    3. **Maintain Clarity**: Use clear and professional language. Ensure the summary is easy to understand.
    4. **Structure the Output**: Organize the summary into distinct paragraphs for Introduction, Methodology, Results, and Conclusion.

    ### Text to Summarize:
    """ + "\n\n".join(retrieved_chunks)

    with st.spinner("Generating summary..."):
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

def generate_research_suggestions(retrieved_chunks, llm):
    """Generates research suggestions from retrieved sections using Mistral."""
    if not retrieved_chunks:
        return "No relevant content found for generating research ideas."

    prompt = """
    ### System Role:
    You are an expert research assistant with a deep understanding of academic research and innovation. Your task is to analyze the provided research content and propose actionable, insightful, and innovative future research ideas.

    ### Instructions:
    1. **Focus on Key Areas**:
       - **Extend the Current Study**: Identify gaps or limitations in the current research and propose how the study could be expanded or improved.
       - **Address Unresolved Challenges**: Highlight any unresolved issues or challenges mentioned in the text and suggest ways to tackle them.
       - **Explore Novel Applications**: Propose new applications or domains where the research findings could be applied.

    2. **Be Specific and Practical**:
       - Provide clear, actionable ideas that are grounded in the provided content.
       - Avoid vague or overly broad suggestions.

    3. **Structure the Output**:
       - Present at least 3 research ideas, each with a brief explanation (1-2 sentences) of its rationale and potential impact.
       - Format the output as a numbered list for clarity.

    ### Text to Analyze:
    """ + "\n\n".join(retrieved_chunks)

    with st.spinner("Generating research suggestions..."):
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

def process_chat_query(query, llm):
    """Handles user queries and answers based on PDF content."""
    retrieved_chunks = search_faiss(query)
    
    if not retrieved_chunks:
        return "No relevant information found in the document."
    
    prompt = """
    ### System Role:
    You are a highly skilled research assistant. Your task is to answer the user's query based on the provided document sections. Ensure your response is accurate, concise, and directly addresses the query.

    ### Instructions:
    1. **Understand the Query**: Carefully analyze the user's question to identify the key points and intent.
    2. **Use Document Context**: Base your response strictly on the provided document sections. Do not add external information or assumptions.
    3. **Be Concise and Clear**: Provide a clear and concise answer. Avoid unnecessary details or repetition.
    4. **Cite Relevant Sections**: If applicable, reference specific parts of the document to support your answer.

    ### Query:
    {query}

    ### Relevant Document Sections:
    """ + "\n\n".join(retrieved_chunks) + """

    ### Response:
    """

    with st.spinner("Processing your question..."):
        response = llm.invoke([HumanMessage(content=prompt.format(query=query))])
        return response.content

def generate_podcast_content(text, podcast_duration, groq_client):
    """Generates intro, Q&A pairs, and outro using Groq API."""
    # Calculate number of questions based on duration (2 minutes per Q&A pair)
    num_questions = max(1, int(podcast_duration // 2))
    
    with st.spinner("Generating podcast introduction..."):
        # Generate Introduction
        intro_prompt = f"""Create a engaging podcast introduction for a research paper. Include:
        1. Warm greeting
        2. Brief context about the research topic
        3. What listeners can expect
        Keep it conversational and under 4 sentences. Paper content: {text[:1000]}"""
        
        intro = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": intro_prompt}],
            temperature=0.7,
        ).choices[0].message.content

    with st.spinner(f"Generating {num_questions} Q&A pairs..."):
        # Generate Q&A Pairs
        qa_prompt = f"""Generate {num_questions} podcast-style Q&A pairs from this research paper. Follow these rules:
        1. Questions should be curious and engaging
        2. Answers should be concise (1-2 short paragraphs)
        3. Use everyday language and examples
        4. Maintain natural flow between questions
        5. Format EXACTLY as: Question: [text]\nAnswer: [text]
        
        Paper content: {text}"""
        
        qa_content = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": qa_prompt}],
            temperature=0.8,
        ).choices[0].message.content

        qa_pairs = []
        for qa in qa_content.split("\n\n"):
            lines = qa.split("\n")
            if len(lines) >= 2:
                question = lines[0].replace("Question: ", "").strip()
                answer = lines[1].replace("Answer: ", "").strip()
                qa_pairs.append(QAFormat(question=question, answer=answer))

    with st.spinner("Generating podcast conclusion..."):
        # Generate Outro
        outro_prompt = f"""Create a podcast closing segment that includes:
        1. Thank you message
        2. Key takeaway
        3. Call to engage (e.g., follow for more content)
        Keep it under 3 sentences and conversational."""
        
        outro = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": outro_prompt}],
            temperature=0.7,
        ).choices[0].message.content

    return intro, qa_pairs, outro

def text_to_speech(intro, qa_pairs, outro, elevenlabs_client):
    """Converts text segments into podcast audio."""
    temp_dir = tempfile.mkdtemp()
    audio_segments = []
    
    try:
        with st.spinner("Generating audio for introduction..."):
            # Generate Introduction
            intro_audio = elevenlabs_client.generate(
                text=intro,
                voice="Rachel",
                model="eleven_multilingual_v2"
            )
            intro_path = os.path.join(temp_dir, "temp_intro.mp3")
            save(intro_audio, intro_path)
            audio_segments.append(AudioSegment.from_mp3(intro_path))
        
        # Generate Q&A
        for idx, qa in enumerate(qa_pairs):
            with st.spinner(f"Generating audio for Q&A pair {idx+1}/{len(qa_pairs)}..."):
                q_audio = elevenlabs_client.generate(
                    text=qa.question,
                    voice="Rachel",
                    model="eleven_multilingual_v2"
                )
                a_audio = elevenlabs_client.generate(
                    text=qa.answer,
                    voice="Adam",
                    model="eleven_multilingual_v2"
                )
                
                q_path = os.path.join(temp_dir, f"temp_q{idx}.mp3")
                a_path = os.path.join(temp_dir, f"temp_a{idx}.mp3")
                
                save(q_audio, q_path)
                save(a_audio, a_path)
                
                q_segment = AudioSegment.from_mp3(q_path)
                a_segment = AudioSegment.from_mp3(a_path)
                silence = AudioSegment.silent(duration=800)
                audio_segments.append(q_segment + silence + a_segment + silence)

        with st.spinner("Generating audio for conclusion..."):
            # Generate Outro
            outro_audio = elevenlabs_client.generate(
                text=outro,
                voice="Rachel",
                model="eleven_multilingual_v2"
            )
            outro_path = os.path.join(temp_dir, "temp_outro.mp3")
            save(outro_audio, outro_path)
            audio_segments.append(AudioSegment.from_mp3(outro_path))
        
        # Combine all segments
        output_path = os.path.join(temp_dir, "research_podcast.mp3")
        with st.spinner("Combining audio segments..."):
            podcast_audio = sum(audio_segments)
            podcast_audio.export(output_path, format="mp3")
        
        # Read the file as binary data
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        
        return audio_bytes
    finally:
        # Ensure cleanup happens even if there's an error
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            st.warning(f"Failed to clean up temporary files: {e}")

def get_binary_file_downloader_html(bin_data, file_label='File', file_name='download.mp3'):
    """Generates HTML code for file download link."""
    b64 = base64.b64encode(bin_data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{file_label}</a>'

# Main app
def main():
    st.set_page_config(
        page_title="Research Paper Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()
    
    # Title and description
    st.title("Research Paper Assistant")
    st.subheader("Upload a research paper and interact with its content")
    
    # Sidebar for upload only (no API key inputs)
    with st.sidebar:
        st.header("Upload Research Paper")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("Process Paper"):
                with st.spinner("Processing PDF..."):
                    # Extract text from PDF
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    
                    if pdf_text:
                        st.session_state.pdf_content = pdf_text
                        st.session_state.pdf_name = uploaded_file.name
                        
                        # Chunk the text
                        st.session_state.chunked_texts = chunk_text(pdf_text)
                        
                        # Create FAISS index
                        embedding_model = load_embedding_model()
                        st.session_state.faiss_index, _ = create_faiss_index(
                            st.session_state.chunked_texts, 
                            embedding_model
                        )
                        
                        st.success(f"PDF '{uploaded_file.name}' processed successfully!")
                    else:
                        st.error("Failed to extract text from the PDF")
        
        # Show currently loaded paper
        if st.session_state.pdf_name:
            st.success(f"Currently loaded: {st.session_state.pdf_name}")

    # Check if a paper is loaded
    if not st.session_state.pdf_content:
        st.warning("Please upload and process a research paper to continue.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Summarize", 
        "💡 Research Suggestions", 
        "💬 Chat with Paper",
        "🎙️ Generate Podcast"
    ])
    
    with tab1:
        st.header("Paper Summary")
        if st.button("Generate Summary"):
            # Get relevant chunks
            query = "Summarize the research paper"
            retrieved_chunks = search_faiss(query)
            
            # Generate summary
            llm = load_llm()
            summary = summarize_retrieved_chunks(retrieved_chunks, llm)
            st.markdown(summary)
    
    with tab2:
        st.header("Research Suggestions")
        if st.button("Generate Research Ideas"):
            # Get relevant chunks
            query = "Suggest future research directions"
            retrieved_chunks = search_faiss(query)
            
            # Generate suggestions
            llm = load_llm()
            suggestions = generate_research_suggestions(retrieved_chunks, llm)
            st.markdown(suggestions)
    
    with tab3:
        st.header("Chat with Paper")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])
        
        # Input for new question
        question = st.chat_input("Ask a question about the paper...")
        if question:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.chat_message("user").write(question)
            
            # Generate response
            llm = load_llm()
            response = process_chat_query(question, llm)
            
            # Add assistant message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    
    with tab4:
        st.header("Generate Research Podcast")
        
        # Podcast settings
        podcast_duration = st.slider(
            "Podcast Duration (minutes)", 
            min_value=4, 
            max_value=20, 
            value=10, 
            step=2
        )
        
        if st.button("Generate Podcast"):
            try:
                # Initialize clients
                groq_client = init_groq_client()
                elevenlabs_client = init_elevenlabs_client()
                
                # Limit text to first 7000 chars for podcast generation
                limited_text = st.session_state.pdf_content[:7000]
                
                # Generate podcast content
                intro, qa_pairs, outro = generate_podcast_content(
                    limited_text, 
                    podcast_duration,
                    groq_client
                )
                
                # Display podcast script
                with st.expander("View Podcast Script"):
                    st.subheader("Introduction")
                    st.write(intro)
                    
                    st.subheader("Q&A Segments")
                    for i, qa in enumerate(qa_pairs):
                        st.write(f"**Q{i+1}:** {qa.question}")
                        st.write(f"**A{i+1}:** {qa.answer}")
                        st.write("---")
                    
                    st.subheader("Conclusion")
                    st.write(outro)
                
                # Generate audio
                audio_bytes = text_to_speech(intro, qa_pairs, outro, elevenlabs_client)
                st.session_state.podcast_file = audio_bytes
                
                # Display audio player
                st.audio(audio_bytes, format="audio/mp3")
                
                # Download link
                st.markdown(
                    get_binary_file_downloader_html(
                        audio_bytes, 
                        'Download Podcast', 
                        f"research_podcast_{st.session_state.pdf_name.replace('.pdf', '')}.mp3"
                    ), 
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"Error generating podcast: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()