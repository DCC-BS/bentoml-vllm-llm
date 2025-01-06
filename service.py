import os
import uuid
from argparse import Namespace
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated
from dotenv import load_dotenv

import fastapi

openai_api_app = fastapi.FastAPI()

load_dotenv()

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.3-70B-Instruct")
TOOL_CALL_PARSER = os.getenv("TOOL_CALL_PARSER", "llama3_json")
ENABLE_TOOL_CALL_PARSER = os.getenv("ENABLE_TOOL_CALL_PARSER", True)
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", 8196))

TIMEOUT = int(os.getenv("TIMEOUT", 300))

GPU_COUNT = int(os.getenv("GPU_COUNT", 1))
ENABLE_PREFIX_CACHING = os.getenv("ENABLE_PREFIX_CACHING", True)
KV_CACHE_TYPE = os.getenv("KV_CACHE_TYPE", "fp8")
MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", 256))

@bentoml.mount_asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="vllm-llm",
    traffic={
        "timeout": TIMEOUT,
    }
)
class VLLM:

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        import vllm.entrypoints.openai.api_server as vllm_api_server

        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_MODEL_LEN,
            speculative_model=None, # disable speculative decoding
            kv_cache_dtype=KV_CACHE_TYPE,
            tensor_parallel_size=GPU_COUNT,
            enable_prefix_caching=True,
            max_num_seqs=MAX_NUM_SEQS,
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        OPENAI_ENDPOINTS = [
            ["/chat/completions", vllm_api_server.create_chat_completion, ["POST"]],
            ["/completions", vllm_api_server.create_completion, ["POST"]],
            ["/models", vllm_api_server.show_available_models, ["GET"]],
        ]

        for route, endpoint, methods in OPENAI_ENDPOINTS:
            openai_api_app.add_api_route(
                path=route,
                endpoint=endpoint,
                methods=methods,
            )

        model_config = self.engine.engine.get_model_config()
        args = Namespace()
        args.model = MODEL_ID
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.response_role = "assistant"
        args.served_model_name = None
        args.chat_template = None
        args.lora_modules = None
        args.prompt_adapters = None
        args.request_logger = None
        args.disable_log_stats = True
        args.return_tokens_as_token_ids = False
        args.enable_tool_call_parser = ENABLE_TOOL_CALL_PARSER
        args.enable_auto_tool_choice = True
        args.tool_call_parser = TOOL_CALL_PARSER
        args.enable_prompt_tokens_details = False

        vllm_api_server.init_app_state(
            self.engine, model_config, openai_api_app.state, args
        )

    @bentoml.api
    async def generate(
            self,
            user_prompt: str = "Explain superconductors like I'm five years old",
            system_prompt: Optional[str] = SYSTEM_PROMPT,
            max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
