<div align="center">
    <h1 align="center">Self-host Language Model with vLLM and BentoML</h1>
</div>

This is a BentoML project, that uses [vLLM](https://vllm.ai), a high-throughput and memory-efficient inference engine, to deploy 
language models.


## Prerequisites

- If you want to test the Service locally, we recommend you use an Nvidia GPU with at least 16G VRAM.
- Gain access to the model in [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) if model requires it.
- Install the package manager uv ([docs](https://docs.astral.sh/uv/))
- Python 3.12 is recommended

## Install dependencies

```bash
git clone https://github.com/DCC-BS/bentoml-vllm-llm.git
cd bentoml-vllm-llm

uv sync

export HF_TOKEN=<your-api-key>
```

## Configurations

The following options can be configured in .env file:

- `HF_TOKEN`: Hugging Face API token.
- `MODEL_ID`: The model ID of the Hugging Face model. Default to `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`.
- `MAX_TOKENS`: The maximum number of tokens to generate. Default to 1024.
- `MAX_MODEL_LEN`: Model context length. If unspecified, will be automatically derived from the model config. Default to 8196.
- `TIMEOUT`: The timeout for the inference in seconds. Default to 300.
- `CONCURRENCY`: The number of concurrent requests to the inference server. Default to 256.
- `GPU_COUNT`: The number of GPUs to use. Default to 1. For Models that require vRAM > 48GB, set this to 2 to use tensor parallel inference.
- `KV_CACHE_TYPE`: The type of the key-value cache. Default to `fp8`. Note: According to this [issue](https://github.com/triton-lang/triton/issues/4319), fp8 cache is not supported on a consumer-GPU (rx8000).
- `MAX_NUM_SEQS`: Maximum number of sequences per iteration. Can be used to reduce the number of concurrent requests in a batch. Defaults to 256.


## Run the BentoML Service

The BentoML Service is defined in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
bentoml build
bentoml serve bentoml-vllm-llm:latest

2024-01-18T07:51:30+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:VLLM" listening on http://localhost:3000 (Press CTRL+C to quit)
INFO 01-18 07:51:40 model_runner.py:501] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 01-18 07:51:40 model_runner.py:505] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode.
INFO 01-18 07:51:46 model_runner.py:547] Graph capturing finished in 6 secs.
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain superconductors like I'\''m five years old",
  "tokens": null
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Explain superconductors like I'm five years old",
        tokens=None
    )
    for response in response_generator:
        print(response)
```

</details>

<details>

<summary>OpenAI-compatible endpoints</summary>

This Service uses the `vllm.entrypoints.openai.api_server` package to set up OpenAI-compatible endpoints (`chat/completions` and `completions`). This means your client can interact with the backend Service (in this case, the VLLM class) as if they were communicating directly with OpenAI's API. This package does not affect your BentoML Service code, and you can use it for other LLMs as well.

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
client.models.list()

chat_completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {
            "role": "user",
            "content": "Explain superconductors like I'm five years old"
        }
    ],
    stream=True,
)
for chunk in chat_completion:
    # Extract and print the content of the model's reply
    print(chunk.choices[0].delta.content or "", end="")
```

These OpenAI-compatible endpoints also support [vLLM extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters). For example, you can force the chat completion output a JSON object by using the `guided_json` parameters:

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
client.models.list()

json_schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    }
}

chat_completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    extra_body=dict(guided_json=json_schema),
)
print(chat_completion.choices[0].message.content)  # will return something like: {"city": "Paris"}
```

All supported extra parameters are listed in [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters).
