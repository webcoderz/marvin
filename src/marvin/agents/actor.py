import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pydantic_ai

import marvin
from marvin.engine.thread import Thread, get_thread
from marvin.prompts import Template
from marvin.utilities.asyncio import run_sync

if TYPE_CHECKING:
    from marvin.agents.team import Team


@dataclass(kw_only=True)
class Actor:
    id: uuid.UUID = field(
        default_factory=uuid.uuid4,
        metadata={"description": "Unique identifier for this actor"},
    )

    instructions: str | None = field(
        default=None, metadata={"description": "Instructions for the actor"}
    )

    prompt: str | Path

    def get_delegates(self) -> list["Actor"] | None:
        """
        A list of actors that this actor can delegate to.
        """
        return None

    def get_agentlet(
        self,
        result_types: list[type],
        tools: list[Callable[..., Any]] | None = None,
        **kwargs,
    ) -> pydantic_ai.Agent[Any, Any]:
        raise NotImplementedError("Subclass must implement get_agentlet")

    def start_turn(self):
        """
        Called when the actor starts its turn.
        """
        pass

    def end_turn(self):
        """
        Called when the actor ends its turn.
        """
        pass

    def get_tools(self) -> list[Callable[..., Any]]:
        """
        A list of tools that this actor can use during its turn.
        """
        return []

    def get_end_turn_tools(self) -> list[type["marvin.engine.end_turn_tools.EndTurn"]]:
        """
        A list of `EndTurn` tools that this actor can use to end its turn.
        """
        return []

    def get_prompt(self) -> str:
        return Template(source=self.prompt).render()

    async def say_async(self, message: str, thread: Thread | str | None = None):
        thread = get_thread(thread)
        if message:
            await thread.add_user_message_async(message=message)
        return await marvin.run_async("Respond to the user.", agent=self, thread=thread)

    def say(self, message: str, thread: Thread | str | None = None):
        return run_sync(self.say_async(message, thread))

    def as_team(self) -> "Team":
        raise NotImplementedError(
            "Subclass must implement as_team in order to be properly orchestrated."
        )
