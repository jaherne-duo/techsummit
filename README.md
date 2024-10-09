# Techsummit RAG Demo

## Requirements

- 5 GBs or more of free disk space (1 for the docker image, 4 for the model)
- 8GBs or more of memory dedicated to Docker
- Stop trustedpath containers

## Prework

Run the following two commands to download the container and version 3.1 of llama

- `docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama`
- `docker exec -it ollama ollama run llama3.1`

After running the second command, you should see output like the following:

```txt
success
>>> Send a message (/? for help)
```

If you see this, you're all set! Type `/bye` to get back to your command prompt.

## Installation

pip commands go here
