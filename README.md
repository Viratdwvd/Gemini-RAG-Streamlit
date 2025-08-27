# Gemini RAG (Streamlit) — LangChain + FAISS
In my project, I used Python + Streamlit since it allowed me to rapidly prototype both the frontend and backend together, which is great for AI projects. If I wanted a more production-ready app, I would separate concerns: React for the frontend (modern UI/UX) and Node.js or Python FastAPI for the backend, which would connect to Ollama and Chroma. The tradeoff is that Streamlit is simpler but less flexible in UI, while React/Node offers more scalability and customization but requires maintaining a split codebase.

Setup, run `python ingest.py` then `streamlit run app.py`. See full instructions inside.
