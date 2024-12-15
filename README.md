# [Basic Project] Implementing a Chatbot with External Blog Information

## 📋 Project Overview

This project demonstrates how to create a **LangChain-based Retrieval-Augmented Generation (RAG)** chatbot using **Streamlit**. The chatbot fetches and summarizes information from an external blog about the "All-in Coding Challenge" winners.

## 🎯 Objectives

- **RAG Integration**: Configure the RAG system to use the external data source: `https://spartacodingclub.kr/blog/all-in-challenge_winner`.
- **LLM Model**: Use the `gpt-4o-mini` model to generate concise answers to user queries.

## 📦 Key Features

1. **Data Crawling and Information Extraction**:
   - Extract award titles, creators, project descriptions, and tech stacks from the external blog.

2. **RAG System**:
   - Embed the extracted data and perform similarity searches to retrieve the most relevant content for user queries.

3. **Streamlit UI**:
   - A simple chatbot interface where users can input queries and receive summarized responses.

## 🛠️ Requirements

- **API Key**: Configure your `OPENAI_API_KEY` in a `.env` file.
- **Model Configuration**: Use `gpt-4o-mini` for language generation and vector embeddings for similarity search.

## 🎥 Demonstration Guide

The accompanying demo video showcases:
1. Running the `app_rag.py` file using Streamlit.
2. Interacting with the chatbot through an external URL.
3. Asking the chatbot, "Summarize the winners of the ALL-in Coding Challenge," and receiving a detailed response.

## 📁 Submission Files

- `app_rag.py` file (contains the chatbot code).
- A demonstration video (showcasing the chatbot in action).
