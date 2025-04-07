import streamlit as st
import validators
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader

# --- Streamlit Page Setup ---
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Paste a URL below and get a smart summary!")

# --- Sidebar: Groq API Key ---
with st.sidebar:
    groq_api_key = st.text_input("🔑 Groq API Key", type="password")

# --- Main Input: URL ---
input_url = st.text_input("Enter YouTube or Website URL", placeholder="Paste the link here...")

# --- Prompt Template ---
prompt_template = """
Provide a summary of the following content in 300 words:

Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# --- YouTube Transcript Function ---
def get_youtube_transcript_text(youtube_url):
    # Normalize youtu.be links
    if "youtu.be" in youtube_url:
        youtube_url = youtube_url.replace("youtu.be/", "youtube.com/watch?v=")

    parsed_url = urlparse(youtube_url)
    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    if not video_id:
        raise ValueError("Could not extract video ID from the YouTube URL.")
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return Document(page_content=text)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        raise ValueError(f"Transcript unavailable for this video: {e}")

# --- Summarize Button ---
if st.button("✨ Summarize the Content"):
    if not groq_api_key.strip() or not input_url.strip():
        st.error("Please provide both the Groq API Key and a valid URL.")
    elif not validators.url(input_url):
        st.error("Invalid URL format. Please paste a full YouTube or website URL.")
    else:
        try:
            with st.spinner("🔍 Loading content and generating summary..."):
                # Load content
                if "youtube.com" in input_url or "youtu.be" in input_url:
                    doc = get_youtube_transcript_text(input_url)
                    docs = [doc]
                else:
                    loader = UnstructuredURLLoader(
                        urls=[input_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                    docs = loader.load()

                # Load LLM and summarize
                llm = ChatGroq(model="Llama3-8b-8192", groq_api_key=groq_api_key)
                chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt)
                summary = chain.run(docs)

                st.success("✅ Summary generated:")
                st.write(summary)

        except Exception as e:
            st.error("🚨 Error occurred while processing the URL.")
            st.exception(e)

# --- Footer ---
st.markdown("---")
st.markdown("Crafted with ❤️ using [LangChain](https://www.langchain.com) + [Groq](https://groq.com)")
