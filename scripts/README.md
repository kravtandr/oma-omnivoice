# Docker transfer scripts

Утилиты для переноса Docker-образов из docker-compose на сервер без интернета.

Два способа: через **приватный registry** или через **tar.gz архивы**.

---

## Предварительная настройка insecure registry

Если registry без TLS, на **обоих** хостах (откуда пушим и куда пулим) нужно один раз добавить его в доверенные:

```json
// /etc/docker/daemon.json
{
  "insecure-registries": ["192.168.1.100:5000"]
}
```

```bash
sudo systemctl restart docker
```

---

## Способ 1: Registry

### Поднять registry (если ещё нет)

```bash
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

### Шаг 1 — push с машины с интернетом

```bash
# Образы уже собраны
./scripts/registry_push.sh 192.168.1.100:5000

# Пересобрать и запушить
./scripts/registry_push.sh 192.168.1.100:5000 docker-compose.yml --build

# Через переменную окружения
REGISTRY=192.168.1.100:5000 BUILD=1 ./scripts/registry_push.sh
```

### Шаг 2 — pull на целевом сервере

```bash
./scripts/registry_pull.sh 192.168.1.100:5000

# С нестандартным compose-файлом
./scripts/registry_pull.sh 192.168.1.100:5000 docker-compose.prod.yml
```

После этого `docker compose up -d` работает без изменений в compose-файле — образы ретагируются обратно в оригинальные имена.

---

## Способ 2: Архивы (USB / SCP)

### Шаг 1 — экспорт на машине с образами

```bash
# Сохранить в ./docker-images/
./scripts/archive_export.sh

# Пересобрать и сохранить
BUILD=1 ./scripts/archive_export.sh

# Указать другую папку и compose-файл
./scripts/archive_export.sh docker-compose.prod.yml /mnt/usb/images
```

Создаётся папка с файлами вида `omnivoice-api_local.tar.gz` и `manifest.txt`.

### Шаг 2 — перенести файлы на сервер

```bash
scp -r ./docker-images user@server:/tmp/docker-images
# или скопировать на USB и подключить на сервере
```

### Шаг 3 — импорт на целевом сервере

```bash
./scripts/archive_import.sh /tmp/docker-images
```

После этого `docker compose up -d` работает сразу.

---

## Параметры скриптов

| Скрипт | Аргументы | Переменные окружения |
|---|---|---|
| `registry_push.sh` | `[REGISTRY] [COMPOSE_FILE] [--build\|--no-build]` | `REGISTRY`, `COMPOSE_FILE`, `BUILD` |
| `registry_pull.sh` | `[REGISTRY] [COMPOSE_FILE]` | `REGISTRY`, `COMPOSE_FILE` |
| `archive_export.sh` | `[COMPOSE_FILE] [OUTPUT_DIR] [--build\|--no-build]` | `COMPOSE_FILE`, `OUTPUT_DIR`, `BUILD` |
| `archive_import.sh` | `[INPUT_DIR]` | `INPUT_DIR` |

По умолчанию: `COMPOSE_FILE=docker-compose.yml`, `OUTPUT_DIR=./docker-images`, `BUILD=0` (кроме `registry_push.sh` — там `BUILD=1`).
