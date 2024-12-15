import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Configure OpenAI Chat Model
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

# Function to fetch and extract blog content
def fetch_blog_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    main_content = soup.find("section", class_="css-18vt64m")

    content = ""
    if main_content:
        # Find all h2 tags and relevant paragraphs
        award_sections = main_content.find_all("h2")
        
        for h2 in award_sections:
            text = h2.get_text(strip=True)
            # Check if this h2 is an award title (using emoji or specific keywords like 대상, 우수상)
            if any(award in text for award in ["🏆 대상", "🎖️ 우수상", "🏅 입선"]):
                # Append the award title
                content += f"\n\n### {text}\n"
                
                # Get the project name in the following <h2> tag (e.g., [Lexi Note] 언어공부 필기 웹 서비스)
                next_h2 = h2.find_next_sibling("h2")
                if next_h2:
                    project_title = next_h2.get_text(strip=True)
                    content += f"**Project Title:** {project_title}\n"
                
                # Get creator information
                creator_tag = h2.find_next("p")
                if creator_tag and creator_tag.find("strong"):
                    creators = creator_tag.find("strong").get_text(strip=True)
                    content += f"**Creators:** {creators}\n"
                
                # Find and add the description
                description_block = h2.find_next("div", class_="my-callout")
                if description_block:
                    description = description_block.get_text(strip=True)
                    content += f"**Description:** {description}\n"
                
                # Get the tech stack
                tech_stack = []
                for p in h2.find_all_next("p"):
                    if "사용한 기술 스택" in p.get_text():
                        tech_stack.append(p.get_text(strip=True))
                if tech_stack:
                    content += f"**Tech Stack:** {', '.join(tech_stack)}\n"
    else:
        content = "Error: The main content could not be found. Please check the HTML structure."
    
    return content

# Extract and split blog content from URL
url = "https://spartacodingclub.kr/blog/all-in-challenge_winner"
content = fetch_blog_content(url)
documents = [Document(page_content=content)]

# Save embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vector_store = Chroma.from_documents(documents, embeddings, persist_directory="chroma_store")
vector_store.persist()

# Streamlit configuration and chatbot UI
st.title("All-in Coding Challenge RAG Chatbot")

# Initialize session state for saving conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# Interface for user input
if prompt := st.chat_input("Enter your question for the chatbot:"):
    # Display and save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using RAG system
    docs = vector_store.similarity_search(prompt, k=3)  # Use top 3 related documents for summarization
    summarized_content = "\n".join([doc.page_content for doc in docs])
    
    question_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Summarize the following context and answer the question: {context} Question: {question}"
    )
    formatted_prompt = question_template.format(context=summarized_content, question=prompt)
    
    # Generate model response
    answer = llm([HumanMessage(content=formatted_prompt)])
    with st.chat_message("assistant"):
        st.markdown(answer.content)
    st.session_state.messages.append({"role": "assistant", "content": answer.content})

    # Save conversation history to a file
    with open("conversation_log.txt", "a") as f:
        f.write(f"User: {prompt}\nChatbot: {answer.content}\n\n")
