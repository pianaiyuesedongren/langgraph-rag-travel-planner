import pytest
from gaode.agents.supervisor import _first_user_text, supervisor_node
from langchain_core.messages import AIMessage, HumanMessage


def test_latest_human_message_is_used() -> None:
    state = {
        "messages": [
            HumanMessage(content="第一次问题"),
            AIMessage(content="第一次回答"),
            HumanMessage(content="最新问题"),
            AIMessage(content="上游 Agent 输出"),
        ]
    }
    assert _first_user_text(state) == "最新问题"


@pytest.mark.asyncio
async def test_rule_supervisor_extracts_constraints() -> None:
    result = await supervisor_node(
        {"messages": [HumanMessage(content="2天北京历史文化游，预算2000元，带父母，饮食清淡")]}
    )
    constraints = result["constraints"]
    assert constraints.city == "北京"
    assert constraints.days == 2
    assert constraints.budget == 2000
    assert "老人" in constraints.travelers
    assert "历史" in constraints.preferences
