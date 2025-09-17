import os
import re
import random
from glob import glob
from typing import List, Dict, Tuple

import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# --- Конфигурация ---
PERSIST_DIR = os.path.abspath("chroma_db")
COLLECTION_NAME = "terraria_kb"
KB_DIR = "knowledge_base"

# 🔗 Актуальный endpoint для YandexGPT (2025)
YANDEX_ENDPOINT = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


def parse_header_body(text: str) -> Tuple[Dict[str, str], str]:
    """
    Парсит текст на шапку (метаданные) и тело.
    Возвращает словарь метаданных и текст.
    """
    parts = re.split(r"\n---\n\n", text, maxsplit=1)
    header, body = (parts[0], parts[1]) if len(parts) == 2 else (None, text)
    meta = {}
    if header:
        for line in header.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    return meta, body


def few_shots_from_kb(kb_dir: str = KB_DIR, n: int = 2) -> List[Tuple[str, str]]:
    """
    Случайно выбирает n файлов из knowledge_base и создаёт примеры Q/A.
    Используется как few-shot контекст в промпте.
    """
    files = sorted(glob(os.path.join(kb_dir, "*.txt")))
    shots = []
    if not files:
        return shots
    picks = random.sample(files, k=min(n, len(files)))
    for path in picks:
        try:
            with open(path, encoding="utf-8") as f:
                txt = f.read()
        except Exception as e:
            print(f"[WARN] Пропущен файл для few-shot: {path} — {e}")
            continue
        meta, body = parse_header_body(txt)
        title = meta.get("Title", os.path.basename(path).rsplit(".", 1)[0])
        sentences = re.split(r"(?<=[.!?])\s+", body.strip())
        answer_text = " ".join(sentences[:2]).strip()[:600]
        question = f"Кто или что такое «{title}»?"
        answer = (
            f"{answer_text}\n\n"
            "Источники:\n"
            f"- {meta.get('Title', title)} — {meta.get('Source', '')}"
        )
        shots.append((question, answer))
    return shots


def format_context(hits: Dict) -> str:
    """
    Форматирует результаты поиска в виде строк с номерами [CTX1], [CTX2]...
    """
    lines = []
    docs = hits["documents"][0]
    metadatas = hits["metadatas"][0]
    for i, (doc, meta) in enumerate(zip(docs, metadatas), start=1):
        title = meta.get("title", "Unknown")
        source = meta.get("source", "")
        lines.append(
            f"[CTX{i}] Title: {title}\n"
            f"Source: {source}\n"
            f"Excerpt:\n{doc}\n"
        )
    return "\n".join(lines)


def init_embedder() -> SentenceTransformer:
    """
    Загружает модель BAAI/bge-m3 для генерации эмбеддингов.
    """
    print("Загрузка модели эмбеддингов BAAI/bge-m3...")
    return SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)


def init_vector_store():
    """
    Подключается к локальной ChromaDB и возвращает коллекцию.
    """
    print("Подключение к векторной базе данных...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        return client.get_collection(COLLECTION_NAME)
    except ValueError:
        raise RuntimeError(
            f"Коллекция '{COLLECTION_NAME}' не найдена. "
            "Убедитесь, что вы запускали build_index.py."
        )


def init_llm_yandex():
    """
    Настраивает функцию chat() для обращения к YandexGPT через REST API.
    Поддерживает аутентификацию через API-ключ или IAM-токен.
    """
    load_dotenv()

    folder_id = os.getenv("YANDEX_FOLDER_ID")
    api_key = os.getenv("YANDEX_API_KEY")
    iam_token = os.getenv("YANDEX_IAM_TOKEN")
    model_name = os.getenv("YANDEX_MODEL", "yandexgpt-lite")

    if not folder_id:
        raise RuntimeError("YANDEX_FOLDER_ID не задан в .env")
    if not (api_key or iam_token):
        raise RuntimeError("Требуется YANDEX_API_KEY или YANDEX_IAM_TOKEN")

    # 🛠 Формат modelUri: gpt://{folder_id}/{model}
    model_uri = f"gpt://{folder_id}/{model_name}"

    # 🔐 Заголовки
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Api-Key {api_key}" if api_key else f"Bearer {iam_token}",
    }

    def chat(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 800) -> str:
        """
        Отправляет запрос к YandexGPT и возвращает ответ.
        """
        payload = {
            "modelUri": model_uri,
            "completionOptions": {
                "temperature": temperature,
                "maxTokens": str(max_tokens),
                "stream": False,
            },
            "messages": messages,
        }

        try:
            response = requests.post(
                YANDEX_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=90
            )
            response.raise_for_status()
            data = response.json()

            # ✅ Путь к тексту в ответе
            return data["result"]["alternatives"][0]["message"]["text"].strip()

        except requests.exceptions.HTTPError as e:
            status = response.status_code
            error_text = response.text
            if status == 401:
                print("[ОШИБКА 401] Ошибка аутентификации:")
                print("  • Проверьте корректность YANDEX_API_KEY или YANDEX_IAM_TOKEN")
                print("  • Убедитесь, что ключ активен и имеет права на доступ к YandexGPT")
            elif status == 403:
                print("[ОШИБКА 403] Доступ запрещён:")
                print("  • Убедитесь, что ваш аккаунт привязан к платёжному аккаунту")
                print("  • Проверьте, что сервисному аккаунту назначена роль 'ai.languageModels.user'")
            return f"[HTTP {status}] Ошибка: {error_text}"

        except requests.exceptions.RequestException as e:
            return f"[Сетевая ошибка] {e}"

        except KeyError:
            return f"[Ошибка разбора] Неожиданный формат ответа от API:\n{response.text}"

        except Exception as e:
            return f"[Неизвестная ошибка] {e}"

    return chat


def build_messages(question: str, context_block: str, fewshots: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """
    Собирает системный промпт, few-shot примеры и вопрос.
    """
    system_prompt = (
        "Ты русскоязычный помощник по базе знаний. "
        "Используй только предоставленный контекст. Если сведений не хватает, скажи об этом. "
        "Думай пошагово внутренне, но в ответе давай только вывод и краткое обоснование. "
        "Формат ответа:\n"
        "- Краткий ответ (2–6 предложений).\n"
        "- Основание: 1–3 пункта из контекста.\n"
        "- Источники: Title + URL."
    )

    examples_text = ""
    if fewshots:
        parts = []
        for i, (q, a) in enumerate(fewshots, 1):
            parts.append(f"Пример {i}\nQ: {q}\nA: {a}")
        examples_text = "\n\n".join(parts)

    user_prompt = (
        f"Контекст (используй только эти фрагменты):\n{context_block}\n\n"
        f"{examples_text}\n\n"
        f"Теперь ответь на вопрос:\nQ: {question}\nA:"
    )

    return [
        {"role": "system", "text": system_prompt},
        {"role": "user", "text": user_prompt},
    ]


def answer_loop():
    """
    Основной цикл: ввод вопроса → поиск → генерация → вывод.
    """
    print("Загрузка модели эмбеддингов...")
    embedder = init_embedder()

    print("Подключение к векторной БД...")
    col = init_vector_store()

    print("Настройка YandexGPT...")
    chat = init_llm_yandex()

    print("\nRAG-бот (YandexGPT) готов. Пиши вопрос (пустая строка — выход).")
    while True:
        try:
            user_input = input("> ").strip()
            if not user_input:
                print("Выход.")
                break

            # Генерация эмбеддинга запроса
            query_embedding = embedder.encode([user_input], normalize_embeddings=True)[0].tolist()

            # Поиск в Chroma
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["metadatas", "documents", "distances"],
            )

            if not results["documents"] or not results["documents"][0]:
                print("Ничего не найдено в контексте. Уточните вопрос.\n" + "-" * 60)
                continue

            # Формирование контекста
            context_block = format_context(results)

            # Few-shot примеры
            fewshot_examples = few_shots_from_kb(KB_DIR, n=2)

            # Сборка сообщений
            messages = build_messages(user_input, context_block, fewshot_examples)

            # Генерация ответа
            answer = chat(messages, temperature=0.2, max_tokens=700)

            # Вывод
            print("\n" + answer.strip() + "\n" + "-" * 60)

        except KeyboardInterrupt:
            print("\n\nВыход по Ctrl+C.")
            break
        except Exception as e:
            print(f"\n[Ошибка]: {e}\n" + "-" * 60)


if __name__ == "__main__":
    answer_loop()