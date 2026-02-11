"""Multimodal content: combine text, images, and blobs in a single message."""

from lmctx import Context, InMemoryBlobStore, Part

# BlobStore holds binary data; Context references it via BlobReference.
store = InMemoryBlobStore()
ctx = Context(blob_store=store)

# Store an image in the BlobStore
image_data = b"\x89PNG\r\n\x1a\n (fake image content)"
ref = store.put(image_data, media_type="image/png", kind="image")

# .user() accepts str, a single Part, or a list of Parts
ctx = ctx.user(
    [
        Part(type="text", text="What's in this image?"),
        Part(type="image", blob=ref),
    ]
)

ctx = ctx.assistant("I see a landscape with mountains and a lake.")

# Inspect the multimodal user message
user_msg = ctx.messages[0]
print(f"User message has {len(user_msg.parts)} parts:")
for i, part in enumerate(user_msg.parts):
    if part.text:
        print(f"  [{i}] type={part.type}, text={part.text!r}")
    elif part.blob:
        print(f"  [{i}] type={part.type}, kind={part.blob.kind}, size={part.blob.size} bytes")

# Retrieve the image bytes back (SHA-256 verified on every get)
blob = user_msg.parts[1].blob
assert blob is not None
retrieved = store.get(blob)
print(f"\nRetrieved {len(retrieved)} bytes, sha256={blob.sha256[:16]}...")
