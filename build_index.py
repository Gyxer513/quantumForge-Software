import os
import re
from glob import glob
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb

# --- Настройки ---
PERSIST_DIR = os.path.abspath("chroma_db")  # Абсолютный путь — критично!
KB_DIR = "knowledge_base"
COLLECTION_NAME = "terraria_kb"

# ВАЖНО: Убедитесь, что этот путь одинаков в build_index.py и query_index.py

# Для отладки: переключитесь на более лёгкую модель (раскомментируйте строку ниже)
MODEL_NAME = "BAAI/bge-m3"  # Медленнее, но качественнее
# MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Быстрее, размер эмбеддинга 384

EMBEDDING_BATCH_SIZE = 32  # Можно уменьшить до 16, если не хватает RAM
CHUNK_MAX_WORDS = 250
CHUNK_OVERLAP = 60


def parse_doc(path: str) -> tuple[str, str, str]:
    """Парсит файл: извлекает Title, Source и тело текста."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать файл {path}: {e}")

    parts = re.split(r"\n---\n\n", txt, maxsplit=1)
    header = parts[0] if len(parts) > 0 else ""
    body = parts[1] if len(parts) == 2 else txt

    title_match = re.search(r"^Title:\s*(.+)$", header, flags=re.MULTILINE)
    source_match = re.search(r"^Source:\s*(.+)$", header, flags=re.MULTILINE)

    title = title_match.group(1).strip() if title_match else "Unknown"
    source = source_match.group(1).strip() if source_match else ""

    return title, source, body


def chunk_text_words(text: str, max_words: int = 250, overlap: int = 60) -> List[Dict]:
    """Разбивает текст на чанки по словам с перекрытием."""
    words = text.split()
    chunks = []
    start = 0
    total = len(words)

    while start < total:
        end = min(start + max_words, total)
        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "start_word": start,
                "end_word": end
            })

        if end >= total:
            break
        start = max(0, end - overlap)

    return chunks


def main():
    print("Запуск сборки векторной базы знаний...")
    print(f"Путь к ChromaDB: {PERSIST_DIR}")
    print(f"Источники: {KB_DIR}/")
    print(f"Коллекция: {COLLECTION_NAME}")

    # Проверка существования директории knowledge_base
    if not os.path.exists(KB_DIR):
        print(f"Ошибка: папка '{KB_DIR}' не найдена.")
        return

    # Поиск .txt файлов
    files = sorted(glob(os.path.join(KB_DIR, "*.txt")))
    if not files:
        print(f"Нет файлов в '{KB_DIR}/'. Сначала запустите apply_replacements.py.")
        return
    print(f"Найдено {len(files)} файлов для индексации.")

    # Загрузка модели
    print(f"Загрузка модели: '{MODEL_NAME}'... (может занять время при первом запуске)")
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        print(f"Модель '{MODEL_NAME}' успешно загружена.")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Совет: проверьте подключение к интернету. Модель скачивается с Hugging Face.")
        return

    # Подготовка ChromaDB
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Удалена старая коллекция: {COLLECTION_NAME}")
    except Exception:
        pass  # Коллекции не было — нормально

    collection = client.create_collection(COLLECTION_NAME)
    print(f"Коллекция '{COLLECTION_NAME}' создана в {PERSIST_DIR}")

    # Сбор данных
    all_texts: List[str] = []
    all_ids: List[str] = []
    all_metas: List[Dict] = []

    for path in files:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        try:
            title, source, body = parse_doc(path)
        except Exception as e:
            print(f"[WARN] Пропущен файл {path}: {e}")
            continue

        chunks = chunk_text_words(body, max_words=CHUNK_MAX_WORDS, overlap=CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            all_texts.append(ch["text"])
            all_ids.append(f"{doc_id}:{i}")
            all_metas.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "title": title,
                "source": source,
                "path": path,
                "start_word": ch["start_word"],
                "end_word": ch["end_word"]
            })

    print(f"Подготовлено: {len(all_texts)} чанков для эмбеддингов.")
    print("Считаем эмбеддинги... Это может занять несколько минут.")

    # Генерация эмбеддингов
    try:
        embeddings = model.encode(
            all_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True  # Покажет прогресс-бар — сигнал активности
        )
    except Exception as e:
        print(f"Ошибка при генерации эмбеддингов: {e}")
        return

    # Сохранение в ChromaDB порциями
    BATCH_SIZE = 1000
    print(f"Сохраняем в ChromaDB (порциями по {BATCH_SIZE})...")
    for i in range(0, len(all_texts), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(all_texts))
        collection.add(
            ids=all_ids[i:end_idx],
            documents=all_texts[i:end_idx],
            embeddings=[emb.tolist() for emb in embeddings[i:end_idx]],
            metadatas=all_metas[i:end_idx]
        )

    print("Загрузка в Chroma завершена.")
    print("Индекс построен и сохранён!")
    print(f"Путь: {PERSIST_DIR}")
    print(f"Коллекция: {COLLECTION_NAME}")
    print(f"Чанков: {len(all_texts)}")


if __name__ == "__main__":
    main()