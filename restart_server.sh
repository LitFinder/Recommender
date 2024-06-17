#!/bin/bash

# Stop the FastAPI server (assuming it's run with uvicorn and managed by a process manager like systemd or supervisor)
pkill -f 'uvicorn.*api:app'

# Start the FastAPI server
nohup uvicorn api:app --host 0.0.0.0 --port 8000 &
