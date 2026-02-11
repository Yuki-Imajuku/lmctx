"""Basic usage: build and query an immutable conversation log."""

from lmctx import Context

# Build an append-only conversation log.
# Every method returns a *new* Context; the original is never modified.
ctx = Context()
ctx = ctx.user("What is the capital of France?")
ctx = ctx.assistant("The capital of France is Paris.")
ctx = ctx.user("What about Germany?")
ctx = ctx.assistant("The capital of Germany is Berlin.")

# Iterate over messages in chronological order
for message in ctx:
    print(f"[{message.role}] {message.parts[0].text}")

# Query the last message (any role)
last = ctx.last()
assert last is not None
print(f"\nLast message: {last.parts[0].text}")

# Query the last message by role
last_user = ctx.last(role="user")
assert last_user is not None
print(f"Last user message: {last_user.parts[0].text}")

# Demonstrate immutability
ctx2 = ctx.user("One more question")
print(f"\nOriginal: {len(ctx)} messages")
print(f"Branched: {len(ctx2)} messages")
