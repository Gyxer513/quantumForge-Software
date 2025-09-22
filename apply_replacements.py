import os
import re
import csv
import json


IN_DIR = "data/clean"
OUT_DIR = "knowledge_base"


def load_terms_map(path="terms_map.json"):
    """
    Загружает словарь замен из JSON-файла и возвращает его вместе со скомпилированными регулярными выражениями.
    Ключи сортируются по длине (сначала самые длинные), чтобы избежать частичных совпадений.
    """
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)

    # Сортируем по длине ключа (длинные — первыми)
    items = sorted(m.items(), key=lambda kv: len(kv[0]), reverse=True)

    # Компилируем паттерны: границы слов, без учёта регистра
    patterns = [
        (re.compile(rf"(?<!\w){re.escape(k)}(?!\w)", flags=re.IGNORECASE | re.UNICODE), v)
        for k, v in items
    ]
    return m, patterns


def replace_all(text, patterns):
    """
    Применяет все замены из списка patterns к тексту.
    """
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


def slugify(s):
    """
    Преобразует строку в безопасное имя файла: удаляет спецсимволы, заменяет пробелы на подчёркивания.
    """
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)  # Удаляем не-буквы, не-цифры, не пробелы, не дефисы
    s = re.sub(r"\s+", "_", s.strip())  # Заменяем множественные пробелы на одно подчёркивание
    return s or "doc"  # Если пусто — возвращаем "doc"


def derive_new_title(orig_title, display_title, terms_map, patterns):
    """
    Определяет новый заголовок:
    1. По точному совпадению (игнорируя регистр).
    2. По замене терминов в display_title.
    """
    # 1. Проверка прямого совпадения
    for k, v in terms_map.items():
        if k.lower() == orig_title.lower() or k.lower() == display_title.lower():
            return v

    # 2. Пробуем применить замены к отображаемому заголовку
    candidate = replace_all(display_title, patterns)
    return candidate


def process_file(clean_path, url, display_title, orig_title, terms_map, patterns):
    """
    Обрабатывает один файл: читает, изменяет заголовок и тело, сохраняет в OUT_DIR.
    Возвращает путь к новому файлу и новый заголовок.
    """
    with open(clean_path, encoding="utf-8") as f:
        text = f.read()

    # Разделяем на шапку и тело по разделителю "---"
    parts = re.split(r"\n---\n\n", text, maxsplit=1)
    header_block = parts[0] if parts else ""
    body = parts[1] if len(parts) == 2 else text

    new_title = derive_new_title(orig_title, display_title, terms_map, patterns)

    # Формируем новую шапку
    new_header = (
        f"Title: {new_title}\n"
        f"Source: {url}\n"
        f"License: CC BY-SA (Fandom)\n"
        f"---\n\n"
    )

    # Применяем замены к телу
    new_body = replace_all(body, patterns)

    # Собираем финальный текст
    out_text = new_header + new_body

    # Генерируем имя файла
    out_name = slugify(new_title) + ".txt"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    return out_path, new_title


def main():
    """
    Основная функция: загружает маппинг терминов, обрабатывает каждый файл из sources.csv,
    сохраняет результаты и обновлённый CSV.
    """
    terms_map, patterns = load_terms_map("terms_map.json")

    # Чтение входного CSV
    rows_in = []
    with open("sources.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_in = list(reader)

    rows_out = []
    for row in rows_in:
        clean_path = row["clean_path"]
        url = row["url"]
        display_title = row.get("display_title") or row["original_title"]
        orig_title = row["original_title"]

        if not os.path.exists(clean_path):
            print(f"[WARN] Файл не найден: {clean_path}")
            continue

        out_path, new_title = process_file(
            clean_path, url, display_title, orig_title, terms_map, patterns
        )
        rows_out.append({
            **row,
            "new_title": new_title,
            "final_path": out_path
        })
        print(f"OK: {display_title} → {new_title}")

    # Сохраняем обновлённый CSV
    output_csv = "sources_mapped.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "original_title",
            "display_title",
            "url",
            "clean_path",
            "new_title",
            "final_path"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Готово. Результат записан в {output_csv}")


if __name__ == "__main__":
    main()