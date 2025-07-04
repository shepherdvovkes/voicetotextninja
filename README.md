# Voice Input Console Recorder (OpenAI Whisper)

Консольный диктовщик для Mac/Linux (Python 3.9+)  
**Без VU-Meter** — фокус на удобном управлении, быстрой транскрипции, автоматическом сохранении текста и аудио.

## Возможности

- Мгновенная запись и распознавание голоса (русский/английский, автоматом)
- Использует OpenAI Whisper API (cloud)
- Детектирует тишину, фрагментирует аудио автоматически (пауза >3 сек или 10 сек макс)
- Панель текстовой диктовки, debug-лог, статус/подсказки (на Rich)
- Все фрагменты сохраняются как `.wav` и `.txt` (одна папка)
- Управление с клавиатуры: Enter — стоп, Ctrl+C — выход

## Установка

1. Клонируй репозиторий/помести файлы в директорию проекта.
2. Установи зависимости:
    ```bash
    pip install sounddevice numpy webrtcvad openai python-dotenv rich
    ```

3. Получи OpenAI API key и положи в `.env`:
    ```
    OPENAI_API_KEY=sk-...
    ```

4. Запусти диктовщик:
    ```bash
    python consolerecorder.py
    ```

## Как пользоваться

- **Enter** — начать запись.  
- Говори фразами до 10 сек. Если замолчал на 3+ сек — автостоп и распознавание.
- Распознанный текст и аудиофрагмент сохраняются автоматически.
- **Enter** — начать следующий фрагмент, **Ctrl+C** — выйти.
- После завершения вся расшифровка появится в синей панели.

## Файлы

- `consolerecorder_no_vu.py` — основной код
- `.env` — файл с ключом OpenAI
- `requirements.txt` (можно сгенерировать `pip freeze > requirements.txt`)
- Сохраняемые аудио/текстовые фрагменты (`.wav`, `.txt`)

## Зависимости

- Python 3.9+
- sounddevice
- numpy
- webrtcvad
- openai
- python-dotenv
- rich

## Замечания

- Работает и на Mac, и на Linux (на Windows не тестировалось)
- Если не работает микрофон — проверь права на доступ и выбранное устройство ввода
- Поддержка только русского и английского (автоматический выбор Whisper)

---

**Вопросы, доработки или новые фичи — пиши!**
