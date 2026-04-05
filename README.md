# OmniVoice TTS

HTTP API и Gradio UI для синтеза речи на базе [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) — диффузионной TTS-модели от Xiaomi/k2-fsa с поддержкой 600+ языков.

## Возможности

- **Voice Clone** — клонирование голоса по референсному аудио (3–10 с)
- **Voice Design** — задание атрибутов голоса текстовой инструкцией (`female, low pitch, british accent`)
- **Auto Voice** — автоматический выбор голоса моделью
- **Streaming TTS** — потоковый синтез длинных текстов по чанкам предложений
- **Non-verbal теги** — `[laughter]`, `[sigh]` и другие прямо в тексте
- **Gradio UI** — интерактивный интерфейс в браузере
- **REST API** — multipart и JSON эндпоинты, Swagger на `/docs`

## Требования

- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker + Docker Compose

## Быстрый старт

### 1. Скачать веса модели

```bash
python3 scripts/download_omnivoice.py
# Веса сохраняются в ../omnivoice-model/ (смонтированы в контейнер как /model)
```

### 2. Запустить

```bash
docker compose up --build
```

- Gradio UI: http://localhost:8080/gradio
- API docs (Swagger): http://localhost:8080/docs
- Health check: http://localhost:8080/health

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `OMNIVOICE_MODEL` | `k2-fsa/OmniVoice` | Путь к весам или HuggingFace repo id. Используйте `/model` для предзагруженных весов |
| `OMNIVOICE_NUM_STEP` | `12` | Шаги диффузии. Меньше — быстрее, больше — качественнее (рекомендуется 16–32) |
| `OMNIVOICE_PRELOAD_ASR` | `false` | Загрузить Whisper при старте (больше VRAM, быстрее первый clone без `ref_text`) |
| `OMNIVOICE_DEVICE` | auto | `cuda` / `cpu` / `mps` |
| `OMNIVOICE_GRADIO_PATH` | `/gradio` | Путь монтирования Gradio UI |
| `OMNIVOICE_GRADIO_ROOT_PATH` | — | Root path за reverse proxy |
| `HF_ENDPOINT` | — | Зеркало HuggingFace, например `https://hf-mirror.com` |

## API

### Синтез (multipart)

```bash
# Auto voice → WAV
curl -X POST http://localhost:8080/v1/generate \
  -F "text=Hello, world!" \
  -o output.wav

# Voice clone
curl -X POST http://localhost:8080/v1/generate \
  -F "text=Текст для синтеза." \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Текст референса." \
  -o output.wav

# Voice design
curl -X POST http://localhost:8080/v1/generate \
  -F "text=Hello!" \
  -F "instruct=female, high pitch, russian accent" \
  -o output.wav
```

### Синтез (JSON)

```bash
curl -X POST http://localhost:8080/v1/generate/json \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "response_format": "wav"
  }' \
  -o output.wav
```

### Потоковый синтез

Текст разбивается на группы предложений, аудио отдаётся по мере готовности каждого чанка — не ожидая конца всего текста. Формат ответа: streaming WAV (заголовок с `size=0xFFFFFFFF` + PCM16).

```bash
# Воспроизведение на лету через ffplay
curl -s -X POST http://localhost:8080/v1/generate/stream \
  -F "text=Первое предложение. Второе предложение. Третье предложение." \
  -F "sentences_per_chunk=1" \
  | ffplay -i pipe:0

# Сохранение в файл
curl -X POST http://localhost:8080/v1/generate/stream \
  -F "text=Длинный текст для синтеза..." \
  -F "sentences_per_chunk=2" \
  -F "ref_audio=@reference.wav" \
  -o stream_output.wav
```

### Параметры запросов

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `text` | string | обязательный | Текст синтеза. Поддерживает `[laughter]`, `[sigh]` и подсказки произношения |
| `ref_audio` | file | — | Референсное аудио для клонирования (WAV, MP3 и др.) |
| `ref_text` | string | — | Транскрипт референса. Без него запустится Whisper ASR |
| `instruct` | string | — | Инструкция для Voice Design |
| `language` | string | auto | Язык синтеза (например `Russian`, `English`) |
| `sentences_per_chunk` | int | 1 | Только для `/stream`: предложений на чанк |
| `num_step` | int | 12 | Шаги диффузии (4–64) |
| `guidance_scale` | float | 2.0 | CFG scale |
| `speed` | float | 1.0 | Скорость речи (0.7–1.3) |
| `duration` | float | — | Фиксированная длительность в секундах (переопределяет `speed`) |
| `denoise` | bool | true | Шумоподавление |
| `postprocess_output` | bool | true | Удаление длинных пауз из результата |
| `response_format` | `wav`/`json` | `wav` | Для `/v1/generate/json`: `json` вернёт base64 |

## Non-verbal теги и произношение

Теги вставляются прямо в текст:

```
Hello [laughter] how are you?
Он сказал [sigh] и замолчал.
```

Подсказки произношения (английские CMU-фонемы в скобках, китайская пиньинь) — см. [карточку модели](https://huggingface.co/k2-fsa/OmniVoice).

## Перенос на сервер без интернета

Подробнее: [`scripts/README.md`](scripts/README.md)

### Через registry

```bash
# На машине с интернетом
./scripts/registry_push.sh 192.168.1.100:5000

# На целевом сервере
./scripts/registry_pull.sh 192.168.1.100:5000
docker compose up -d
```

### Через архивы (USB / SCP)

```bash
# Экспорт
./scripts/archive_export.sh

# Перенести ./docker-images/ на сервер, затем:
./scripts/archive_import.sh /path/to/docker-images
docker compose up -d
```

## Структура проекта

```
.
├── api/
│   ├── main.py              # FastAPI приложение (эндпоинты, стриминг)
│   └── requirements-api.txt
├── OmniVoice/               # Исходники модели (submodule / копия)
│   └── omnivoice/
│       ├── cli/demo.py      # Gradio UI (4 вкладки включая Streaming TTS)
│       └── utils/text.py    # Разбивка текста на предложения
├── scripts/
│   ├── download_omnivoice.py
│   ├── registry_push.sh     # Push образов в insecure registry
│   ├── registry_pull.sh     # Pull образов с registry
│   ├── archive_export.sh    # Экспорт образов в tar.gz
│   ├── archive_import.sh    # Импорт образов из tar.gz
│   └── README.md
├── omnivoice-model/         # Предзагруженные веса (не в git)
├── Dockerfile
└── docker-compose.yml
```
