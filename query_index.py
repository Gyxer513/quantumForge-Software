import chromadb
from sentence_transformers import SentenceTransformer

# Настройки
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "terraria_kb"

def main():
    # Инициализация клиента и коллекции
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    col = client.get_collection(COLLECTION_NAME)

    # Загрузка модели эмбеддингов
    model = SentenceTransformer(
        "BAAI/bge-m3",
        trust_remote_code=True
    )

    print("Поиск по базе знаний. Введите запрос (или пустую строку для выхода)")

    while True:
        # Получение запроса от пользователя
        q = input("\nЗапрос: ").strip()

        if not q:
            break

        # Вычисление эмбеддинга запроса
        q_emb = model.encode(
            [q],
            normalize_embeddings=True
        )[0].tolist()

        # Поиск по базе
        res = col.query(
            query_embeddings=[q_emb],
            n_results=5,
            include=["metadatas", "documents", "distances"]
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        print("\nТоп-результаты:")

        for i, (doc, meta, dist) in enumerate(
            zip(docs, metas, dists),
            start=1
        ):
            title = meta.get("title", "Unknown")
            source = meta.get("source", "")

            print(f"{i}) {title} | score={1 - dist:.3f}")

            if source:
                print(f"   Source: {source}")

            print(f"   Snippet: {doc[:200].replace('\\n', ' ')}...")

        print("-" * 60)

if __name__ == "__main__":
    main()
