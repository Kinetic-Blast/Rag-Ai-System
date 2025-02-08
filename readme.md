# Retrieval-Augmented Generation (RAG) AI System

This project implements a Retrieval-Augmented Generation (RAG) system that enhances AI-generated responses with relevant information retrieved from a local vector database. The system is designed to process and store document embeddings efficiently, enabling context-aware AI interactions.

## Features

- **Document Processing**: Extracts text from PDFs and stores embeddings in an SQLite vector database.  
- **Efficient Retrieval**: Queries relevant information based on user input to improve AI responses.  
- **AI Integration**: Uses `google/gemma-2b-it` to generate detailed and structured responses.  
- **Token Management**: Maintains context within token limits using a sliding window approach.  
- **Auto-Add System**: Automatically processes and embeds new files for seamless updates.  

## How It Works

1. Documents are processed and converted into embeddings.  
2. User queries trigger a retrieval of relevant data from the vector database.  
3. Retrieved information is formatted and provided as context to the AI model.  
4. The AI generates a response, integrating both retrieved knowledge and general reasoning.  

## Issues

- The system works, but `gemma-2b-it` produces relatively short responses and may need to be replaced with a more capable model.  
- Some tuning is required, but as a first implementation, it functions well. Future improvements will focus on enhancing response quality and retrieval accuracy.  

