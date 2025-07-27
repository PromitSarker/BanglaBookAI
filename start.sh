#!/bin/bash
set -e

# Start FastAPI on port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Give it a moment
sleep 2

# Start Streamlit on port 8501
exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501