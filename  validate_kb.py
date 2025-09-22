import os
import re
import json


def main():
    """
    Проверяет, остались ли в обработанных файлах из knowledge_base
    оригинальные термины из terms_map.json (до замены).
    Если такие термины найдены — выводит список файлов и проблемных слов.
    """
    # Загружаем маппинг терминов
    try:
        with open("terms_map.json", "r", encoding="utf-8") as f:
            terms = json.load(f)
    except FileNotFoundError:
        print("Ошибка: файл terms_map.json не найден.")
        return
    except json.JSONDecodeError as e:
        print(f"Ошибка чтения JSON: {e}")
        return

    originals = list(terms.keys())

    bad = []  # Список файлов, где найдены старые термины

    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        print(f"Ошибка: папка '{kb_dir}' не существует.")
        return

    # Обходим все .txt файлы в knowledge_base
    for fn in os.listdir(kb_dir):
        if not fn.endswith(".txt"):
            continue

        path = os.path.join(kb_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                txt = f.read()
        except Exception as e:
            print(f"[WARN] Не удалось прочитать файл {path}: {e}")
            continue

        # Ищем любой из оригинальных терминов как целое слово (с учётом регистра)
        found = False
        for k in originals:
            if re.search(
                rf"(?<!\w){re.escape(k)}(?!\w)",
                txt,
                flags=re.IGNORECASE | re.UNICODE
            ):
                bad.append((fn, k))
                found = True
                break  # Достаточно одного совпадения

    # Вывод результата
    if bad:
        print(f"⚠️ Найдены оригинальные термины в {len(bad)} файлах:")
        for f, k in bad[:20]:  # Показываем максимум 20 примеров
            print(f" - {f}: '{k}'")
        if len(bad) > 20:
            print(f" ... и ещё {len(bad) - 20} шт.")
        print("\n💡 Решение: добавьте недостающие формы в terms_map.json "
              "и запустите apply_replacements.py снова.")
    else:
        print("✅ OK: оригинальные термины не найдены. Все замены выполнены корректно.")


if __name__ == "__main__":
    main()