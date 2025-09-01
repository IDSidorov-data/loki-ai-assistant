# build_cloner.py
import os
import pprint
from pathlib import Path

# --- НАСТРОЙКИ ---
# 1. Укажите путь к исходной папке вашего проекта.
SOURCE_DIRECTORY = "C:\Projects\loki_project"
# 2. Укажите имя папки, которую будет создавать финальный скрипт.
TARGET_PROJECT_NAME = "loki_project_recreated"
# 3. Имя файла, который будет сгенерирован.
OUTPUT_SCRIPT_NAME = "create_loki_clone.py"

# 4. Папки и файлы, которые нужно исключить из сборки.
EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    "loki.egg-info",
    "models",
    "poetry.lock",
    "build_cloner.py",
    ".env",  # Исключаем сгенерированную папку
}
EXCLUDE_FILES = {
    ".DS_Store",
}
# --- КОНЕЦ НАСТРОЕК ---


def scan_project_structure(root_dir):
    """
    Сканирует структуру проекта и читает содержимое файлов.

    Args:
        root_dir (str): Путь к корневой папке проекта.

    Returns:
        dict: Словарь, представляющий структуру проекта.
    """
    project_structure = {}
    root_path = Path(root_dir).resolve()

    for path in sorted(root_path.rglob("*")):
        # Пропускаем исключенные директории и файлы
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name in EXCLUDE_FILES:
            continue

        relative_path = path.relative_to(root_path)

        if path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
                # Добавляем файл в структуру
                current_level = project_structure
                parts = list(relative_path.parts)
                for part in parts[:-1]:
                    current_level = current_level.setdefault(part, {})
                current_level[parts[-1]] = content
            except Exception as e:
                print(f"Не удалось прочитать файл {path}: {e}. Пропускаю.")

    return project_structure


def create_cloner_script(structure, target_name, output_filename):
    """
    Генерирует Python-скрипт, который воссоздает структуру проекта.

    Args:
        structure (dict): Словарь со структурой проекта.
        target_name (str): Имя корневой папки для воссоздания.
        output_filename (str): Имя генерируемого скрипта.
    """
    # Преобразуем словарь в красивую строку для вставки в код
    structure_str = pprint.pformat(structure, indent=4, width=120)

    # Шаблон для финального скрипта
    script_template = f'''
# -*- coding: utf-8 -*-
"""
Автоматически сгенерированный скрипт для воссоздания проекта LOKI.
Запустите этот скрипт, чтобы создать полную структуру папок и файлов.
"""
import os
from pathlib import Path

# --- СТРУКТУРА ПРОЕКТА И СОДЕРЖИМОЕ ФАЙЛОВ ---
PROJECT_STRUCTURE = {structure_str}

# --- ИМЯ КОРНЕВОЙ ПАПКИ ПРОЕКТА ---
ROOT_DIR_NAME = "{target_name}"


def create_project(root_dir, structure):
    """
    Рекурсивно создает папки и файлы на основе словаря-структуры.
    """
    root_path = Path(root_dir)
    print(f"Создание корневой папки: {{root_path}}")
    root_path.mkdir(exist_ok=True)

    for name, content in structure.items():
        path = root_path / name
        if isinstance(content, dict):  # Если это папка
            print(f"  -> Создание подпапки: {{path}}")
            path.mkdir(exist_ok=True)
            create_project(path, content)
        else:  # Если это файл
            print(f"    -> Запись файла: {{path}}")
            try:
                path.write_text(content, encoding="utf-8")
            except Exception as e:
                print(f"      [ОШИБКА] Не удалось записать файл {{path}}: {{e}}")

def main():
    """
    Основная функция для запуска процесса создания проекта.
    """
    print("--- Начало воссоздания проекта LOKI ---")
    if Path(ROOT_DIR_NAME).exists():
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Папка '{{ROOT_DIR_NAME}}' уже существует.")
        answer = input("Вы хотите продолжить и, возможно, перезаписать файлы? (y/n): ").lower()
        if answer != 'y':
            print("Операция отменена.")
            return
            
    try:
        create_project(ROOT_DIR_NAME, PROJECT_STRUCTURE)
        print("\\n--- Проект успешно воссоздан в папке '{{ROOT_DIR_NAME}}' ---")
    except Exception as e:
        print(f"\\n[КРИТИЧЕСКАЯ ОШИБКА] Произошла ошибка во время создания проекта: {{e}}")

if __name__ == "__main__":
    main()
'''

    # Записываем сгенерированный скрипт в файл
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(script_template.strip())
        print(f"\n[УСПЕХ] Скрипт-упаковщик '{output_filename}' успешно создан!")
    except Exception as e:
        print(f"\n[ОШИБКА] Не удалось создать файл скрипта: {e}")


def main():
    """Главная функция."""
    print(f"Сканирую проект в папке '{SOURCE_DIRECTORY}'...")

    source_path = Path(SOURCE_DIRECTORY)
    if not source_path.is_dir():
        print(f"[ОШИБКА] Исходная директория '{SOURCE_DIRECTORY}' не найдена.")
        print("Пожалуйста, убедитесь, что скрипт запущен из правильного места,")
        print(f"и папка '{SOURCE_DIRECTORY}' находится рядом.")
        return

    project_data = scan_project_structure(SOURCE_DIRECTORY)

    if not project_data:
        print(
            "[ОШИБКА] Не удалось найти файлы в исходной директории. Ничего не сгенерировано."
        )
        return

    create_cloner_script(project_data, TARGET_PROJECT_NAME, OUTPUT_SCRIPT_NAME)


if __name__ == "__main__":
    main()
