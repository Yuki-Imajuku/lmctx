"""Tool calling: build tool_call and tool_result messages."""

from lmctx import Context, Message, Part

ctx = Context()

# 1. User asks a question that needs a tool
ctx = ctx.user("What's the weather in Tokyo?")

# 2. Assistant responds with a tool call
ctx = ctx.append(
    Message(
        role="assistant",
        parts=(
            Part(
                type="tool_call",
                tool_call_id="call_001",
                tool_name="get_weather",
                tool_args={"city": "Tokyo", "unit": "celsius"},
            ),
        ),
    )
)

# 3. Tool result comes back as a "tool" role message
ctx = ctx.append(
    Message(
        role="tool",
        parts=(
            Part(
                type="tool_result",
                tool_call_id="call_001",
                tool_output={"temperature": 22, "condition": "sunny", "humidity": 45},
            ),
        ),
    )
)

# 4. Assistant gives the final answer
ctx = ctx.assistant("It's currently 22 degrees C and sunny in Tokyo with 45% humidity.")

# Inspect the full conversation
print("Conversation:")
for msg in ctx:
    for part in msg.parts:
        if part.type == "text":
            print(f"  [{msg.role}] {part.text}")
        elif part.type == "tool_call":
            print(f"  [{msg.role}] -> {part.tool_name}({part.tool_args})")
        elif part.type == "tool_result":
            print(f"  [{msg.role}] <- {part.tool_output}")

# Multiple tool calls in a single assistant message
ctx2 = Context().user("Compare weather in Tokyo and London")
ctx2 = ctx2.append(
    Message(
        role="assistant",
        parts=(
            Part(type="tool_call", tool_call_id="call_a", tool_name="get_weather", tool_args={"city": "Tokyo"}),
            Part(type="tool_call", tool_call_id="call_b", tool_name="get_weather", tool_args={"city": "London"}),
        ),
    )
)

print(f"\nParallel tool calls: {len(ctx2.messages[1].parts)} calls in one message")
