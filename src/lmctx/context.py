"""Context: the append-only conversation log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeVar, overload

from lmctx.blobs import BlobStore, InMemoryBlobStore
from lmctx.types import Cursor, Message, Part, Role, Usage

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

PipeResultT = TypeVar("PipeResultT")


def _normalize_content(content: str | Part | Sequence[Part]) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (Part(type="text", text=content),)
    if isinstance(content, Part):
        return (content,)

    parts: list[Part] = []
    for index, item in enumerate(content):
        if not isinstance(item, Part):
            msg = f"content sequence items must be Part instances; got {type(item).__name__} at index {index}."
            raise TypeError(msg)
        parts.append(item)
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class Context:
    """Append-only conversation log.

    By default, every mutation method returns a new Context instance.
    Set ``inplace=True`` to mutate the current snapshot and return ``None``.
    The blob_store is intentionally shared across snapshots so that
    all Context instances from the same chain can resolve the same blobs.
    """

    messages: tuple[Message, ...] = ()
    cursor: Cursor = field(default_factory=Cursor)
    usage_log: tuple[Usage, ...] = ()
    blob_store: BlobStore = field(default_factory=InMemoryBlobStore)
    __hash__ = None

    def __post_init__(self) -> None:
        """Normalize containers so runtime behavior matches type hints."""
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "usage_log", tuple(self.usage_log))

    # --- Internal state transition helper ---

    def _next(
        self,
        *,
        messages: tuple[Message, ...] | None = None,
        cursor: Cursor | None = None,
        usage_log: tuple[Usage, ...] | None = None,
        inplace: bool = False,
    ) -> Context | None:
        """Build the next snapshot or mutate the current one when ``inplace``."""
        next_messages = self.messages if messages is None else messages
        next_cursor = self.cursor if cursor is None else cursor
        next_usage_log = self.usage_log if usage_log is None else usage_log

        if inplace:
            object.__setattr__(self, "messages", next_messages)
            object.__setattr__(self, "cursor", next_cursor)
            object.__setattr__(self, "usage_log", next_usage_log)
            return None

        return Context(
            messages=next_messages,
            cursor=next_cursor,
            usage_log=next_usage_log,
            blob_store=self.blob_store,
        )

    # --- Append operations ---

    @overload
    def append(self, message: Message, *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def append(self, message: Message, *, inplace: Literal[True]) -> None: ...

    @overload
    def append(self, message: Message, *, inplace: bool) -> Context | None: ...

    def append(self, message: Message, *, inplace: bool = False) -> Context | None:
        """Append a Message.

        Returns a new Context by default. If ``inplace=True``, mutates ``self`` and returns ``None``.
        """
        return self._next(messages=(*self.messages, message), inplace=inplace)

    @overload
    def extend(self, messages: Sequence[Message], *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def extend(self, messages: Sequence[Message], *, inplace: Literal[True]) -> None: ...

    @overload
    def extend(self, messages: Sequence[Message], *, inplace: bool) -> Context | None: ...

    def extend(self, messages: Sequence[Message], *, inplace: bool = False) -> Context | None:
        """Append multiple messages at once."""
        if not messages:
            return self._next(inplace=inplace)
        return self._next(messages=(*self.messages, *messages), inplace=inplace)

    @overload
    def user(self, content: str | Part | Sequence[Part], *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def user(self, content: str | Part | Sequence[Part], *, inplace: Literal[True]) -> None: ...

    @overload
    def user(self, content: str | Part | Sequence[Part], *, inplace: bool) -> Context | None: ...

    def user(self, content: str | Part | Sequence[Part], *, inplace: bool = False) -> Context | None:
        """Append a user message."""
        return self.append(Message(role="user", parts=_normalize_content(content)), inplace=inplace)

    @overload
    def assistant(self, content: str | Part | Sequence[Part], *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def assistant(self, content: str | Part | Sequence[Part], *, inplace: Literal[True]) -> None: ...

    @overload
    def assistant(self, content: str | Part | Sequence[Part], *, inplace: bool) -> Context | None: ...

    def assistant(self, content: str | Part | Sequence[Part], *, inplace: bool = False) -> Context | None:
        """Append an assistant message."""
        return self.append(Message(role="assistant", parts=_normalize_content(content)), inplace=inplace)

    @overload
    def with_cursor(self, cursor: Cursor, *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def with_cursor(self, cursor: Cursor, *, inplace: Literal[True]) -> None: ...

    @overload
    def with_cursor(self, cursor: Cursor, *, inplace: bool) -> Context | None: ...

    def with_cursor(self, cursor: Cursor, *, inplace: bool = False) -> Context | None:
        """Return a Context with an updated Cursor."""
        return self._next(cursor=cursor, inplace=inplace)

    @overload
    def with_usage(self, usage: Usage, *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def with_usage(self, usage: Usage, *, inplace: Literal[True]) -> None: ...

    @overload
    def with_usage(self, usage: Usage, *, inplace: bool) -> Context | None: ...

    def with_usage(self, usage: Usage, *, inplace: bool = False) -> Context | None:
        """Return a Context with an appended Usage entry."""
        return self._next(usage_log=(*self.usage_log, usage), inplace=inplace)

    @overload
    def clear(self, *, inplace: Literal[False] = False) -> Context: ...

    @overload
    def clear(self, *, inplace: Literal[True]) -> None: ...

    @overload
    def clear(self, *, inplace: bool) -> Context | None: ...

    def clear(self, *, inplace: bool = False) -> Context | None:
        """Clear messages, cursor, and usage log."""
        return self._next(messages=(), cursor=Cursor(), usage_log=(), inplace=inplace)

    def clone(self) -> Context:
        """Return a shallow clone of the Context snapshot."""
        return Context(
            messages=self.messages,
            cursor=self.cursor,
            usage_log=self.usage_log,
            blob_store=self.blob_store,
        )

    def pipe(
        self,
        func: Callable[..., PipeResultT],
        /,
        *args: object,
        **kwargs: object,
    ) -> PipeResultT:
        """Apply a callable to the Context and return its output."""
        return func(self, *args, **kwargs)

    # --- Query operations ---

    def last(self, *, role: Role | None = None) -> Message | None:
        """Return the last message, optionally filtered by role."""
        for message in reversed(self.messages):
            if role is None or message.role == role:
                return message
        return None

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        """Iterate over messages in chronological order."""
        return iter(self.messages)
