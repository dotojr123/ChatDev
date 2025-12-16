import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from chatdev.chat_chain import ChatChain
from camel.typing import ModelType

@pytest.mark.asyncio
async def test_chat_chain_init(mock_settings):
    # Mocking file operations since ChatChain loads config from files
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        with patch("json.load") as mock_json_load:
            # Setup mock configs
            mock_json_load.side_effect = [
                {"chain": [], "recruitments": [], "web_spider": False, "clear_structure": "False", "gui_design": "False", "git_management": "False", "incremental_develop": "False", "background_prompt": "", "with_memory": "False", "self_improve": "False"}, # ChatChainConfig
                {"DemandAnalysis": {"assistant_role_name": "A", "user_role_name": "B", "phase_prompt": ["p1"]}}, # PhaseConfig
                {"A": ["prompt A"], "B": ["prompt B"], "Chief Executive Officer": ["CEO Prompt"], "Counselor": ["Counselor Prompt"]} # RoleConfig
            ]

            chain = ChatChain(
                config_path="config.json",
                config_phase_path="phase.json",
                config_role_path="role.json",
                task_prompt="Test Task",
                project_name="TestProject",
                org_name="TestOrg",
                model_type=ModelType.GPT_3_5_TURBO
            )

            assert chain.project_name == "TestProject"

            # Mock pre_processing
            chain.pre_processing = AsyncMock()
            chain.execute_chain = AsyncMock()

            await chain.pre_processing()
            await chain.execute_chain()

            chain.pre_processing.assert_awaited_once()
            chain.execute_chain.assert_awaited_once()
