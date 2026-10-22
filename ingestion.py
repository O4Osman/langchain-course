import hashlib
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone

NAMESPACE = "langchain-course"

if __name__ == '__main__':
    print("Ingesting...")
    loader = UnstructuredLoader(file_path="./mediumblog1.txt", chunking_strategy="basic", max_characters=10000,
                                autodetect_encoding=True)
    document = loader.load()
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(f"created {len(chunks)} chunks")

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.index(name=os.environ["PINECONE_INDEX"])

    records = []

    for position, chunk in enumerate(chunks):
        text = chunk.page_content.strip()
        if not text:
            continue

        source = str(
            chunk.metadata.get("source", "mediumblog1.txt")
        )
        title = str(
            chunk.metadata.get("title", "Medium Blog")
        )

        record_id = hashlib.sha256(
            f"{source}:{position}:{text}".encode("utf-8")
        ).hexdigest()

        records.append({
            "_id": record_id,
            "chunk_text": text,
            "title": title,
            "source": source,
            "chunk_number": position,
        })

    if not records:
        print("No non-empty chunks to ingest.")
        # return

    print("Uploading documents...")

    result = index.documents.batch_upsert(
        namespace=NAMESPACE,
        documents=records,
        batch_size=50,
        show_progress=True,
    )

    print(f"Successful: {result.successful_item_count}")
    print(f"Failed: {result.failed_item_count}")

    if result.failed_items:
        print("Some documents failed:")
        for item in result.failed_items:
            print(item)

    print("Ingestion finished.")
