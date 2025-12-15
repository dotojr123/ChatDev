# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import openai
import tiktoken
import os

from camel.typing import ModelType
from chatdev.statistics import prompt_cost
from chatdev.utils import log_visualize
from chatdev.settings import settings

# We assume new API is always available as per requirements
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

class ModelBackend(ABC):
    r"""Base class for different model backends."""

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    async def run_async(self, *args, **kwargs):
        pass

    @abstractmethod
    async def get_embedding_async(self, text: str) -> List[float]:
        pass


class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

        # Initialize clients
        api_key = settings.OPENAI_API_KEY
        base_url = settings.OPENAI_API_BASE

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _calculate_cost(self, usage, model_name):
        return prompt_cost(
            model_name,
            num_prompt_tokens=usage.prompt_tokens,
            num_completion_tokens=usage.completion_tokens
        )

    def _prepare_request(self, messages):
        string = "\n".join([message["content"] for message in messages])
        try:
            encoding = tiktoken.encoding_for_model(self.model_type.value)
            num_prompt_tokens = len(encoding.encode(string))
        except KeyError:
            # Fallback for newer models or unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
            num_prompt_tokens = len(encoding.encode(string))

        gap_between_send_receive = 15 * len(messages)
        num_prompt_tokens += gap_between_send_receive

        num_max_token_map = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }

        # Default to a safe large number if model not found
        num_max_token = num_max_token_map.get(self.model_type.value, 128000)
        num_max_completion_tokens = max(4096, num_max_token - num_prompt_tokens)

        # Create a copy to not mutate shared state
        config = self.model_config_dict.copy()
        # config['max_tokens'] = num_max_completion_tokens

        return config

    def run(self, *args, **kwargs):
        config = self._prepare_request(kwargs.get("messages", []))

        response = self.client.chat.completions.create(
            *args,
            **kwargs,
            model=self.model_type.value,
            **config
        )

        cost = self._calculate_cost(response.usage, self.model_type.value)
        log_visualize(
            "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                response.usage.prompt_tokens, response.usage.completion_tokens,
                response.usage.total_tokens, cost))

        return response

    async def run_async(self, *args, **kwargs):
        config = self._prepare_request(kwargs.get("messages", []))

        response = await self.async_client.chat.completions.create(
            *args,
            **kwargs,
            model=self.model_type.value,
            **config
        )

        cost = self._calculate_cost(response.usage, self.model_type.value)
        log_visualize(
            "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                response.usage.prompt_tokens, response.usage.completion_tokens,
                response.usage.total_tokens, cost))

        return response

    async def get_embedding_async(self, text: str) -> List[float]:
        try:
            # Using small embedding model by default
            response = await self.async_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            log_visualize(f"Error getting embedding: {str(e)}")
            return []


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self._stub_response()

    async def run_async(self, *args, **kwargs):
        return self._stub_response()

    async def get_embedding_async(self, text: str) -> List[float]:
        # Return dummy embedding of size 1536
        return [0.1] * 1536

    def _stub_response(self):
        # Return object that mimics OpenAI response object structure for compatibility
        class Usage:
            prompt_tokens = 10
            completion_tokens = 10
            total_tokens = 20

        class Choice:
            finish_reason = "stop"
            class Message:
                content = "Lorem Ipsum"
                role = "assistant"
            message = Message()

        class Response:
            id = "stub_model_id"
            usage = Usage()
            choices = [Choice()]

        return Response()


class ModelFactory:
    r"""Factory of backend models."""

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        if model_type == ModelType.STUB:
            model_class = StubModel
        else:
            model_class = OpenAIModel

        inst = model_class(model_type, model_config_dict)
        return inst
