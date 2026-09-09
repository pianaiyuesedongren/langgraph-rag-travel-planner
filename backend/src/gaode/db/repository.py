from __future__ import annotations

from sqlalchemy import delete, select

from gaode.db.models import AgentRunRecord, Conversation, MessageRecord, TravelPlanRecord
from gaode.db.session import get_session_factory
from gaode.schemas.travel import TravelResponse


async def append_message(thread_id: str, role: str, content: str) -> bool:
    factory = get_session_factory()
    if factory is None:
        return False
    async with factory() as session, session.begin():
        conversation = await session.get(Conversation, thread_id)
        if conversation is None:
            conversation = Conversation(id=thread_id, title=content[:80] or "新旅行规划")
            session.add(conversation)
        session.add(MessageRecord(conversation_id=thread_id, role=role, content=content))
    return True


async def save_travel_response(question: str, response: TravelResponse) -> bool:
    factory = get_session_factory()
    if factory is None or response.plan is None:
        return False
    async with factory() as session, session.begin():
        conversation = await session.get(Conversation, response.thread_id)
        if conversation is None:
            conversation = Conversation(
                id=response.thread_id,
                title=question[:80] or f"{response.plan.city}旅行规划",
            )
            session.add(conversation)
        session.add(
            TravelPlanRecord(
                id=response.request_id,
                thread_id=response.thread_id,
                question=question,
                city=response.plan.city,
                days=response.plan.days,
                budget=response.plan.constraints.budget,
                answer=response.answer,
                plan_data=response.plan.model_dump(mode="json"),
                generation_meta=response.generation_meta.model_dump(mode="json"),
            )
        )
        session.add_all(
            AgentRunRecord(
                request_id=response.request_id,
                thread_id=response.thread_id,
                step=trace.step,
                message=trace.message,
                event_data=trace.data,
            )
            for trace in response.traces
        )
    return True


async def get_messages(thread_id: str) -> list[dict]:
    factory = get_session_factory()
    if factory is None:
        return []
    async with factory() as session:
        rows = (
            await session.execute(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == thread_id)
                .order_by(MessageRecord.created_at)
            )
        ).scalars()
        return [
            {
                "role": row.role,
                "content": row.content,
                "timestamp": row.created_at.isoformat(),
            }
            for row in rows
        ]


async def delete_conversation(thread_id: str) -> None:
    factory = get_session_factory()
    if factory is None:
        return
    async with factory() as session, session.begin():
        await session.execute(delete(Conversation).where(Conversation.id == thread_id))
