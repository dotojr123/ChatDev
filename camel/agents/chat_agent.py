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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from camel.agents import BaseAgent
from camel.configs import ChatGPTConfig
from camel.messages import ChatMessage, MessageType, SystemMessage
from camel.model_backend import ModelBackend, ModelFactory
from camel.typing import ModelType, RoleType
from camel.utils import (
    get_model_token_limit,
    num_tokens_from_messages,
    openai_api_key_required,
)
from chatdev.utils import log_visualize
from chatdev.settings import settings
from openai.types.chat import ChatCompletion
import structlog

logger = structlog.get_logger()

@dataclass(frozen=True)
class ChatAgentResponse:
    msgs: List[ChatMessage]
    terminated: bool
    info: Dict[str, Any]

    @property
    def msg(self):
        if self.terminated:
            raise RuntimeError("error in ChatAgentResponse, info:{}".format(str(self.info)))
        if len(self.msgs) > 1:
            raise RuntimeError("Property msg is only available for a single message in msgs")
        elif len(self.msgs) == 0:
            if len(self.info) > 0:
                raise RuntimeError("Empty msgs in ChatAgentResponse, info:{}".format(str(self.info)))
            else:
                return None
        return self.msgs[0]


class ChatAgent(BaseAgent):
    def __init__(
            self,
            system_message: SystemMessage,
            memory = None,
            model: Optional[ModelType] = None,
            model_config: Optional[Any] = None,
            message_window_size: Optional[int] = None,
    ) -> None:

        self.system_message: SystemMessage = system_message
        self.role_name: str = system_message.role_name
        self.role_type: RoleType = system_message.role_type
        self.model: ModelType = (model if model is not None else ModelType.GPT_3_5_TURBO)
        self.model_config: ChatGPTConfig = model_config or ChatGPTConfig()
        self.model_token_limit: int = get_model_token_limit(self.model)
        self.message_window_size: Optional[int] = message_window_size
        self.model_backend: ModelBackend = ModelFactory.create(self.model, self.model_config.__dict__)
        self.terminated: bool = False
        self.info: bool = False
        self.init_messages()
        # Deprecated memory usage in Agent, should use global vector store if needed,
        # but keeping reference for compatibility if memory passed is the old object
        self.memory = memory if memory and hasattr(memory, 'memory_data') else None

        # New Vector Store Reference (injected or global)
        # For this refactor, we assume the environment passes the new Qdrant store instance in `memory`
        # if it's not the old legacy one. To be safe, we check type or attributes.
        self.vector_store = None
        if memory and hasattr(memory, 'search'): # Duck typing for VectorStore interface
            self.vector_store = memory

    def reset(self) -> List[MessageType]:
        self.terminated = False
        self.init_messages()
        return self.stored_messages

    def get_info(
            self,
            id: Optional[str],
            usage: Optional[Dict[str, int]],
            termination_reasons: List[str],
            num_tokens: int,
    ) -> Dict[str, Any]:
        return {
            "id": id,
            "usage": usage,
            "termination_reasons": termination_reasons,
            "num_tokens": num_tokens,
        }

    def init_messages(self) -> None:
        self.stored_messages: List[MessageType] = [self.system_message]

    def update_messages(self, message: ChatMessage) -> List[MessageType]:
        self.stored_messages.append(message)
        return self.stored_messages

    async def use_memory_async(self, input_message) -> Optional[str]:
        """
        Retrieves relevant context from the vector store based on input message.
        """
        if not self.vector_store:
            return None

        try:
            # Generate embedding using the model backend
            embedding = await self.model_backend.get_embedding_async(input_message)
            if not embedding:
                return None

            # Search in Vector Store
            results = await self.vector_store.search(query_vector=embedding, limit=3)

            if results:
                # Format results as a string
                context = "\n\nRelevant past memories:\n"
                for res in results:
                    context += f"- {res.content}\n"
                return context

        except Exception as e:
            logger.error("Failed to query memory", error=str(e))

        return None

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    @openai_api_key_required
    async def step(
            self,
            input_message: ChatMessage,
    ) -> ChatAgentResponse:
        messages = self.update_messages(input_message)
        if self.message_window_size is not None and len(
                messages) > self.message_window_size:
            messages = [self.system_message
                        ] + messages[-self.message_window_size:]
        openai_messages = [message.to_openai_message() for message in messages]
        num_tokens = num_tokens_from_messages(openai_messages, self.model)

        output_messages: Optional[List[ChatMessage]]
        info: Dict[str, Any]

        if num_tokens < self.model_token_limit:
            # Async call
            response = await self.model_backend.run_async(messages=openai_messages)

            # OpenAI Object -> ChatMessage
            output_messages = [
                ChatMessage(role_name=self.role_name, role_type=self.role_type,
                            meta_dict=dict(), **dict(choice.message))
                for choice in response.choices
            ]
            info = self.get_info(
                response.id,
                response.usage,
                [str(choice.finish_reason) for choice in response.choices],
                num_tokens,
            )

            if output_messages[0].content.split("\n")[-1].startswith("<INFO>"):
                self.info = True
        else:
            self.terminated = True
            output_messages = []

            info = self.get_info(
                None,
                None,
                ["max_tokens_exceeded_by_camel"],
                num_tokens,
            )

        return ChatAgentResponse(output_messages, self.terminated, info)

    def __repr__(self) -> str:
        return f"ChatAgent({self.role_name}, {self.role_type}, {self.model})"
