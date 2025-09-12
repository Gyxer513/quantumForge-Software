import os
import time
import csv
import json
import re
import requests
from bs4 import BeautifulSoup
import trafilatura

# Базовые настройки
BASE = "https://terraria.fandom.com/ru"
API = BASE + "/api.php"
WIKI = BASE + "/wiki"

# Создание директорий
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)


def fetch_html(title):
    """
    Получение HTML-содержимого страницы через MediaWiki API.

    Args:
        title (str): Название статьи.

    Returns:
        tuple: (html, page_url, display_title) или (None, None, None) при ошибке.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "text|displaytitle",
        "format": "json"
    }
    headers = {"User-Agent": "RAG-edu-bot/1.0"}

    try:
        r = requests.get(API, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        j = r.json()

        if "error" in j or "parse" not in j:
            err = j.get("error", {}).get("info", "unknown error")
            print(f"[WARN] Не удалось распарсить {title}: {err}")
            return None, None, None

        html = j["parse"]["text"]["*"]
        display_title = j["parse"].get("displaytitle", title)
        page_url = f"{WIKI}/{title.replace(' ', '_')}"

        return html, page_url, display_title

    except Exception as e:
        print(f"[ERROR] {title}: {e}")
        return None, None, None


def clean_text_from_html(html):
    """
    Очистка HTML от мусора и нормализация текста.

    Args:
        html (str): HTML-содержимое страницы.

    Returns:
        str: Очищенный текст.
    """
    # Попытка извлечь текст через trafilatura
    txt = trafilatura.extract(html, include_tables=False)
    if not txt:
        soup = BeautifulSoup(html, "html.parser")

        # Удаление нежелательных элементов
        for tag in soup(["script", "style"]):
            tag.decompose()

        selectors_to_remove = [
            ".portable-infobox", ".pi-item", ".toc", ".footer", ".navbox",
            ".reference", ".reflist", ".catlinks", ".mw-editsection"
        ]
        for selector in selectors_to_remove:
            for element in soup.select(selector):
                element.decompose()

        # Извлечение текста
        txt = soup.get_text("\n", strip=True)

    # Нормализация текста
    txt = re.sub(r"\[[0-9]+\]", "", txt)          # убрать сноски [1], [2]
    txt = re.sub(r"[ \t]+", " ", txt)             # лишние пробелы
    txt = re.sub(r"\n{3,}", "\n\n", txt)         # более двух пустых строк → две
    return txt.strip()


def slugify(s):
    """
    Преобразует строку в безопасное имя файла.

    Args:
        s (str): Строка для преобразования.

    Returns:
        str: "slugified" строка.
    """
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    return s or "doc"


def main():
    """Основная функция: загрузка, очистка и сохранение статей."""
    # Чтение списка страниц
    with open("pages.txt", "r", encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip()]

    # Открытие CSV для записи метаданных
    with open("sources.csv", "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=["original_title", "display_title", "url", "clean_path"]
        )
        writer.writeheader()

        for title in titles:
            print(f"Обработка: {title}")
            html, url, display_title = fetch_html(title)
            if not html:
                continue

            # Антиспам задержка
            time.sleep(1.2)

            # Сохранение сырого HTML
            raw_path = f"data/raw/{slugify(title)}.html"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(html)

            # Очистка и сохранение текста
            text = clean_text_from_html(html)
            header = (
                f"Title: {display_title}\n"
                f"Source: {url}\n"
                f"License: CC BY-SA (Fandom)\n"
                f"---\n\n"
            )
            clean_path = f"data/clean/{slugify(title)}.txt"
            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(header + text)

            # Запись в CSV
            writer.writerow({
                "original_title": title,
                "display_title": display_title,
                "url": url,
                "clean_path": clean_path
            })


if __name__ == "__main__":
    main()