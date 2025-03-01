from marvin.engine.events import (
    ActorEndTurnEvent,
    ActorMessageDeltaEvent,
    ActorMessageEvent,
    ActorStartTurnEvent,
    EndTurnToolCallEvent,
    EndTurnToolResultEvent,
    Event,
    OrchestratorEndEvent,
    OrchestratorExceptionEvent,
    OrchestratorStartEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolRetryEvent,
    UserMessageEvent,
)


class Handler:
    def handle(self, event: Event):
        """Handle is called whenever an event is emitted.

        By default, it dispatches to a method named after the event type e.g.
        `self.on_{event_type}(event=event)`.

        The `on_event` method is always called for every event.
        """
        self.on_event(event=event)
        event_type = event.type.replace("-", "_")
        method = getattr(self, f"on_{event_type}", None)
        if method:
            method(event=event)

    def on_event(self, event: Event):
        """Called for every event before specific handlers. Override for global event processing."""
        pass

    def on_user_message(self, event: UserMessageEvent):
        """Handles messages from the user. Called when a user sends input to the system."""
        pass

    def on_actor_message(self, event: ActorMessageEvent):
        """Handles complete messages from actors. Called when an actor produces a full response."""
        pass

    def on_actor_message_delta(self, event: ActorMessageDeltaEvent):
        """Handles streaming updates to actor messages. Called multiple times during message generation."""
        pass

    def on_tool_result(self, event: ToolResultEvent):
        """Handles successful tool execution results. Called when a tool returns output."""
        pass

    def on_tool_retry(self, event: ToolRetryEvent):
        """Handles tool retry events. Called when a tool execution fails and needs to be retried."""
        pass

    def on_tool_call(self, event: ToolCallEvent):
        """Handles tool call events. Called when an actor invokes a tool."""
        pass

    def on_tool_call_delta(self, event: ToolCallDeltaEvent):
        """Handles streaming updates during tool call formation. Called multiple times as tool calls are generated."""
        pass

    def on_end_turn_tool_call(self, event: EndTurnToolCallEvent):
        """Handles special end-of-turn tool calls. Called when actors use tools to signal turn completion."""
        pass

    def on_end_turn_tool_result(self, event: EndTurnToolResultEvent):
        """Handles the final result from an end turn tool."""
        pass

    def on_actor_start_turn(self, event: ActorStartTurnEvent):
        """Handles actor turn initiation. Called when an actor begins its processing turn."""
        pass

    def on_actor_end_turn(self, event: ActorEndTurnEvent):
        """Handles actor turn completion. Called when an actor finishes its processing turn."""
        pass

    def on_orchestrator_start(self, event: OrchestratorStartEvent):
        """Handles orchestrator initialization. Called once when orchestration begins."""
        pass

    def on_orchestrator_end(self, event: OrchestratorEndEvent):
        """Handles normal orchestrator completion. Called when the orchestration finishes successfully."""
        pass

    def on_orchestrator_exception(self, event: OrchestratorExceptionEvent):
        """Handles orchestrator exceptions. Called when an error occurs during orchestration."""
        pass


class AsyncHandler:
    async def handle(self, event: Event):
        """Handle is called whenever an event is emitted.

        By default, it dispatches to a method named after the event type e.g.
        `self.on_{event_type}(event=event)`.

        The `on_event` method is always called for every event.
        """
        await self.on_event(event=event)
        event_type = event.type.replace("-", "_")
        method = getattr(self, f"on_{event_type}", None)
        if method:
            await method(event=event)

    async def on_event(self, event: Event):
        """Called for every event before specific handlers. Override for global event processing."""
        pass

    async def on_user_message(self, event: UserMessageEvent):
        """Handles messages from the user. Called when a user sends input to the system."""
        pass

    async def on_actor_message(self, event: ActorMessageEvent):
        """Handles complete messages from actors. Called when an actor produces a full response."""
        pass

    async def on_actor_message_delta(self, event: ActorMessageDeltaEvent):
        """Handles streaming updates to actor messages. Called multiple times during message generation."""
        pass

    async def on_tool_result(self, event: ToolResultEvent):
        """Handles successful tool execution results. Called when a tool returns output."""
        pass

    async def on_tool_retry(self, event: ToolRetryEvent):
        """Handles tool retry events. Called when a tool execution fails and needs to be retried."""
        pass

    async def on_tool_call(self, event: ToolCallEvent):
        """Handles tool call events. Called when an actor invokes a tool."""
        pass

    async def on_tool_call_delta(self, event: ToolCallDeltaEvent):
        """Handles streaming updates during tool call formation. Called multiple times as tool calls are generated."""
        pass

    async def on_end_turn_tool_call(self, event: EndTurnToolCallEvent):
        """Handles special end-of-turn tool calls. Called when actors use tools to signal turn completion."""
        pass

    async def on_end_turn_tool_result(self, event: EndTurnToolResultEvent):
        """Handles the final result from an end turn tool."""
        pass

    async def on_actor_start_turn(self, event: ActorStartTurnEvent):
        """Handles actor turn initiation. Called when an actor begins its processing turn."""
        pass

    async def on_actor_end_turn(self, event: ActorEndTurnEvent):
        """Handles actor turn completion. Called when an actor finishes its processing turn."""
        pass

    async def on_orchestrator_start(self, event: OrchestratorStartEvent):
        """Handles orchestrator initialization. Called once when orchestration begins."""
        pass

    async def on_orchestrator_end(self, event: OrchestratorEndEvent):
        """Handles normal orchestrator completion. Called when the orchestration finishes successfully."""
        pass

    async def on_orchestrator_exception(self, event: OrchestratorExceptionEvent):
        """Handles orchestrator exceptions. Called when an error occurs during orchestration."""
        pass
