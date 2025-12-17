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
import copy
from typing import Dict, List, Optional, Sequence, Tuple

from camel.agents import (
    ChatAgent,
    TaskPlannerAgent,
    TaskSpecifyAgent,
)
from camel.agents.chat_agent import ChatAgentResponse
from camel.messages import ChatMessage, UserChatMessage
from camel.messages import SystemMessage
from camel.typing import ModelType, RoleType, TaskType, PhaseType
from chatdev.utils import log_arguments, log_visualize


@log_arguments
class RolePlaying:
    def __init__(
            self,
            assistant_role_name: str,
            user_role_name: str,
            critic_role_name: str = "critic",
            task_prompt: str = "",
            assistant_role_prompt: str = "",
            user_role_prompt: str = "",
            user_role_type: Optional[RoleType] = None,
            assistant_role_type: Optional[RoleType] = None,
            with_task_specify: bool = True,
            with_task_planner: bool = False,
            with_critic_in_the_loop: bool = False,
            critic_criteria: Optional[str] = None,
            model_type: ModelType = ModelType.GPT_3_5_TURBO,
            task_type: TaskType = TaskType.AI_SOCIETY,
            assistant_agent_kwargs: Optional[Dict] = None,
            user_agent_kwargs: Optional[Dict] = None,
            task_specify_agent_kwargs: Optional[Dict] = None,
            task_planner_agent_kwargs: Optional[Dict] = None,
            critic_kwargs: Optional[Dict] = None,
            sys_msg_generator_kwargs: Optional[Dict] = None,
            extend_sys_msg_meta_dicts: Optional[List[Dict]] = None,
            extend_task_specify_meta_dict: Optional[Dict] = None,
            background_prompt: Optional[str] = "",
            memory = None,
    ) -> None:
        self.with_task_specify = with_task_specify
        self.with_task_planner = with_task_planner
        self.with_critic_in_the_loop = with_critic_in_the_loop
        self.model_type = model_type
        self.task_type = task_type
        self.memory = memory

        # Task Specify and Planner are still synchronous in initialization for now
        # Ideally they should also be async if they use LLMs
        if with_task_specify:
            task_specify_meta_dict = dict()
            if self.task_type in [TaskType.AI_SOCIETY, TaskType.MISALIGNMENT]:
                task_specify_meta_dict.update(
                    dict(assistant_role=assistant_role_name,
                         user_role=user_role_name))
            if extend_task_specify_meta_dict is not None:
                task_specify_meta_dict.update(extend_task_specify_meta_dict)

            task_specify_agent = TaskSpecifyAgent(
                self.model_type,
                task_type=self.task_type,
                **(task_specify_agent_kwargs or {}),
            )
            # Warning: This is still synchronous
            self.specified_task_prompt = task_specify_agent.step(
                task_prompt,
                meta_dict=task_specify_meta_dict,
            )
            task_prompt = self.specified_task_prompt
        else:
            self.specified_task_prompt = None

        if with_task_planner:
            task_planner_agent = TaskPlannerAgent(
                self.model_type,
                **(task_planner_agent_kwargs or {}),
            )
            self.planned_task_prompt = task_planner_agent.step(
                task_prompt,
                assistant_role_name=assistant_role_name,
                user_role_name=user_role_name,
            )
            task_prompt = f"{task_prompt}\n{self.planned_task_prompt}"
        else:
            self.planned_task_prompt = None

        self.task_prompt = task_prompt

        sys_msg_meta_dicts = [dict(chatdev_prompt=background_prompt, task=task_prompt)] * 2
        if (extend_sys_msg_meta_dicts is None and self.task_type in [TaskType.AI_SOCIETY, TaskType.MISALIGNMENT,
                                                                     TaskType.CHATDEV]):
            extend_sys_msg_meta_dicts = [dict(assistant_role=assistant_role_name, user_role=user_role_name)] * 2
        if extend_sys_msg_meta_dicts is not None:
            sys_msg_meta_dicts = [{**sys_msg_meta_dict, **extend_sys_msg_meta_dict} for
                                  sys_msg_meta_dict, extend_sys_msg_meta_dict in
                                  zip(sys_msg_meta_dicts, extend_sys_msg_meta_dicts)]

        self.assistant_sys_msg = SystemMessage(role_name=assistant_role_name, role_type=RoleType.DEFAULT,
                                               meta_dict=sys_msg_meta_dicts[0],
                                               content=assistant_role_prompt.format(**sys_msg_meta_dicts[0]))
        self.user_sys_msg = SystemMessage(role_name=user_role_name, role_type=RoleType.DEFAULT,
                                          meta_dict=sys_msg_meta_dicts[1],
                                          content=user_role_prompt.format(**sys_msg_meta_dicts[1]))

        self.assistant_agent: ChatAgent = ChatAgent(self.assistant_sys_msg, memory, model_type,
                                                    **(assistant_agent_kwargs or {}), )
        self.user_agent: ChatAgent = ChatAgent(self.user_sys_msg,memory, model_type, **(user_agent_kwargs or {}), )

        self.critic = None

    async def init_chat(self, phase_type: PhaseType = None,
                  placeholders=None, phase_prompt=None):
        if placeholders is None:
            placeholders = {}
        self.assistant_agent.reset()
        self.user_agent.reset()

        content = phase_prompt.format(
            **({"assistant_role": self.assistant_agent.role_name} | placeholders)
        )

        # Async memory retrieval - now enabled
        retrieval_memory = await self.assistant_agent.use_memory_async(content)
        if retrieval_memory!= None:
            placeholders["examples"] = retrieval_memory

        user_msg = UserChatMessage(
            role_name=self.user_sys_msg.role_name,
            role="user",
            content=content
        )
        pseudo_msg = copy.deepcopy(user_msg)
        pseudo_msg.role = "assistant"
        self.user_agent.update_messages(pseudo_msg)

        log_visualize(self.user_agent.role_name,
                      "**[Start Chat]**\n\n[" + self.assistant_agent.system_message.content + "]\n\n" + content)
        return None, user_msg

    def process_messages(
            self,
            messages: Sequence[ChatMessage],
    ) -> ChatMessage:
        if len(messages) == 0:
            raise ValueError("No messages to process.")
        if len(messages) > 1:
             raise ValueError("Got more than one message to process.")

        return messages[0]

    async def step(
            self,
            user_msg: ChatMessage,
            assistant_only: bool,
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        assert isinstance(user_msg, ChatMessage)

        user_msg_rst = user_msg.set_user_role_at_backend()

        # Async Step
        assistant_response = await self.assistant_agent.step(user_msg_rst)

        if assistant_response.terminated or assistant_response.msgs is None:
            return (
                ChatAgentResponse([assistant_response.msgs], assistant_response.terminated, assistant_response.info),
                ChatAgentResponse([], False, {}))

        assistant_msg = self.process_messages(assistant_response.msgs)
        if self.assistant_agent.info:
            return (ChatAgentResponse([assistant_msg], assistant_response.terminated, assistant_response.info),
                    ChatAgentResponse([], False, {}))
        self.assistant_agent.update_messages(assistant_msg)

        if assistant_only:
            return (
                ChatAgentResponse([assistant_msg], assistant_response.terminated, assistant_response.info),
                ChatAgentResponse([], False, {})
            )

        assistant_msg_rst = assistant_msg.set_user_role_at_backend()

        # Async Step
        user_response = await self.user_agent.step(assistant_msg_rst)

        if user_response.terminated or user_response.msgs is None:
            return (ChatAgentResponse([assistant_msg], assistant_response.terminated, assistant_response.info),
                    ChatAgentResponse([user_response], user_response.terminated, user_response.info))
        user_msg = self.process_messages(user_response.msgs)
        if self.user_agent.info:
            return (ChatAgentResponse([assistant_msg], assistant_response.terminated, assistant_response.info),
                    ChatAgentResponse([user_msg], user_response.terminated, user_response.info))
        self.user_agent.update_messages(user_msg)

        return (
            ChatAgentResponse([assistant_msg], assistant_response.terminated, assistant_response.info),
            ChatAgentResponse([user_msg], user_response.terminated, user_response.info),
        )
