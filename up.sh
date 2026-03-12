docker build -t llm-evaluation-ragas -f Dockerfile.dev .
docker run --rm \
    --env-file .env \
    -p 8000:8000 \
    -v "$(pwd):/app" \
    llm-evaluation-ragas