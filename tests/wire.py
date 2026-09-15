"""Realistic wire responses consumed by the installed GoodMem SDK."""

SPACE_ID = "00000000-0000-4000-8000-000000000001"
MEMORY_ID = "00000000-0000-4000-8000-000000000002"
CHUNK_ID = "00000000-0000-4000-8000-000000000003"
EMBEDDER_ID = "00000000-0000-4000-8000-000000000004"
OWNER_ID = "00000000-0000-4000-8000-000000000005"


def common():
    return {
        "createdAt": 1700000000000,
        "updatedAt": 1700000000000,
        "createdById": OWNER_ID,
        "updatedById": OWNER_ID,
    }


def space(**overrides):
    return {
        **common(),
        "spaceId": SPACE_ID,
        "name": "test-space",
        "labels": {},
        "ownerId": OWNER_ID,
        "spaceEmbedders": [
            {
                **common(),
                "spaceId": overrides.get("spaceId", SPACE_ID),
                "embedderId": EMBEDDER_ID,
                "defaultRetrievalWeight": 1.0,
            }
        ],
        **overrides,
    }


def memory(**overrides):
    return {
        **common(),
        "memoryId": MEMORY_ID,
        "spaceId": SPACE_ID,
        "originalContentRef": "",
        "contentType": "text/plain",
        "metadata": {},
        "processingStatus": "PENDING",
        "pageImageStatus": "UNSPECIFIED",
        "pageImageCount": 0,
        **overrides,
    }


def chunk(text="A useful test memory.", **overrides):
    return {
        "retrievedItem": {
            "chunk": {
                "resultSetId": "default",
                "memoryIndex": 0,
                "relevanceScore": -0.9,
                "chunk": {
                    **common(),
                    "chunkId": CHUNK_ID,
                    "memoryId": MEMORY_ID,
                    "chunkText": text,
                    "chunkSequenceNumber": 0,
                    "vectorStatus": "COMPLETED",
                    "startOffset": 0,
                    "endOffset": len(text),
                    "metadata": {},
                    **overrides,
                },
            }
        }
    }
