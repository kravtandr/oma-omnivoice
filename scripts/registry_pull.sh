#!/usr/bin/env bash
# =============================================================================
# registry_pull.sh — pull образов с insecure registry на целевой хост
#
# Использование:
#   ./scripts/registry_pull.sh [REGISTRY] [COMPOSE_FILE]
#
# Примеры:
#   ./scripts/registry_pull.sh 192.168.1.100:5000
#   REGISTRY=192.168.1.100:5000 ./scripts/registry_pull.sh
#   REGISTRY=192.168.1.100:5000 ./scripts/registry_pull.sh docker-compose.prod.yml
#
# После выполнения образы будут доступны локально под оригинальными тегами,
# так что docker compose up будет работать без изменений в compose-файле.
# =============================================================================
set -euo pipefail

REGISTRY="${1:-${REGISTRY:-}}"
COMPOSE_FILE="${2:-${COMPOSE_FILE:-docker-compose.yml}}"

if [[ -z "$REGISTRY" ]]; then
    echo "ОШИБКА: укажите адрес registry (аргумент или переменная REGISTRY)"
    echo "  Пример: $0 192.168.1.100:5000"
    exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "ОШИБКА: файл '$COMPOSE_FILE' не найден"
    exit 1
fi

# --- получение образов из compose-файла --------------------------------------
get_images() {
    docker compose -f "$COMPOSE_FILE" config 2>/dev/null \
        | grep -E '^\s+image:' \
        | awk '{print $2}' \
        | sort -u
}

# --- основная логика ----------------------------------------------------------
echo "========================================"
echo " registry_pull.sh"
echo "========================================"
echo "  Registry    : $REGISTRY"
echo "  Compose     : $COMPOSE_FILE"
echo "========================================"

echo
echo ">> Получение списка образов из $COMPOSE_FILE..."
IMAGES=$(get_images)

if [[ -z "$IMAGES" ]]; then
    echo "ОШИБКА: образы не найдены в $COMPOSE_FILE"
    exit 1
fi

echo "  Найдены образы:"
echo "$IMAGES" | sed 's/^/    - /'

PULLED=()
FAILED=()

while IFS= read -r image; do
    [[ -z "$image" ]] && continue

    # Убираем возможный registry-префикс из оригинального имени
    if [[ "$image" =~ ^[^/]+\.[^/]+/ ]] || [[ "$image" =~ ^[^/]+:[0-9]+/ ]]; then
        bare="${image#*/}"
    else
        bare="$image"
    fi

    remote_name="$REGISTRY/$bare"

    echo
    echo "  Тяну: $remote_name"
    if docker pull "$remote_name"; then
        # Ретаг обратно в оригинальное имя, чтобы compose-файл не трогать
        if [[ "$remote_name" != "$image" ]]; then
            docker tag "$remote_name" "$image"
            echo "  Ретаг: $remote_name → $image"
        fi
        PULLED+=("$image")
        echo "  [ok]"
    else
        FAILED+=("$remote_name")
        echo "  [ОШИБКА]"
    fi
done <<< "$IMAGES"

echo
echo "========================================"
echo " ИТОГ"
echo "========================================"
if [[ ${#PULLED[@]} -gt 0 ]]; then
    echo "  Успешно получено (${#PULLED[@]}):"
    printf '    - %s\n' "${PULLED[@]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  ОШИБКИ (${#FAILED[@]}):"
    printf '    - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "========================================"
echo
echo "  Теперь можно запустить: docker compose -f $COMPOSE_FILE up -d"
