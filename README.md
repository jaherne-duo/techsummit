# Techsummit RAG Demo

## Requirements

- 5 GBs or more of free disk space (1 for the docker image, 4 for the model)
- 8GBs or more of memory dedicated to Docker
- Stop trustedpath containers

## Prework

Run the following three commands to download the container, version 3.1 of llama, and the embedding model.

- `docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`
- `docker exec -it ollama ollama pull llama3.1`
- `docker exec -it ollama ollama pull mxbai-embed-large`

## Installation

pip commands go here
