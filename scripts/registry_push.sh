#!/usr/bin/env bash
# =============================================================================
# registry_push.sh — сборка образов из docker-compose и push в insecure registry
#
# Использование:
#   ./scripts/registry_push.sh [REGISTRY] [COMPOSE_FILE] [--build] [--no-build]
#
# Примеры:
#   ./scripts/registry_push.sh 192.168.1.100:5000
#   ./scripts/registry_push.sh 192.168.1.100:5000 docker-compose.prod.yml
#   REGISTRY=192.168.1.100:5000 ./scripts/registry_push.sh
#   REGISTRY=192.168.1.100:5000 BUILD=1 ./scripts/registry_push.sh
#
# Переменные окружения:
#   REGISTRY     — адрес registry (обязательно, если не передан аргументом)
#   COMPOSE_FILE — путь к compose-файлу (по умолчанию: docker-compose.yml)
#   BUILD        — 1 = пересобрать образы перед push (по умолчанию: 1)
# =============================================================================
set -euo pipefail

# --- аргументы ----------------------------------------------------------------
REGISTRY="${1:-${REGISTRY:-}}"
COMPOSE_FILE="${2:-${COMPOSE_FILE:-docker-compose.yml}}"

# третий аргумент или переменная BUILD
if [[ "${3:-}" == "--no-build" ]]; then
    BUILD=0
elif [[ "${3:-}" == "--build" ]]; then
    BUILD=1
else
    BUILD="${BUILD:-1}"
fi

# --- проверки -----------------------------------------------------------------
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
    # docker compose config раскрывает все переменные и возвращает итоговый конфиг
    docker compose -f "$COMPOSE_FILE" config 2>/dev/null \
        | grep -E '^\s+image:' \
        | awk '{print $2}' \
        | sort -u
}

# --- основная логика ----------------------------------------------------------
echo "========================================"
echo " registry_push.sh"
echo "========================================"
echo "  Registry    : $REGISTRY"
echo "  Compose     : $COMPOSE_FILE"
echo "  Build       : $([[ $BUILD -eq 1 ]] && echo 'да' || echo 'нет')"
echo "========================================"

# Собираем образы
if [[ $BUILD -eq 1 ]]; then
    echo
    echo ">> Сборка образов..."
    docker compose -f "$COMPOSE_FILE" build
fi

# Получаем список образов
echo
echo ">> Получение списка образов из $COMPOSE_FILE..."
IMAGES=$(get_images)

if [[ -z "$IMAGES" ]]; then
    echo "ОШИБКА: образы не найдены в $COMPOSE_FILE"
    exit 1
fi

echo "  Найдены образы:"
echo "$IMAGES" | sed 's/^/    - /'

# Тегируем и пушим
echo
echo ">> Тегирование и push..."
PUSHED=()
FAILED=()

while IFS= read -r image; do
    [[ -z "$image" ]] && continue

    # Имя без тега → берём только имя:тег без registry-префикса
    local_name="$image"
    # Убираем возможный существующий registry-префикс (host:port/...)
    if [[ "$image" =~ ^[^/]+\.[^/]+/ ]] || [[ "$image" =~ ^[^/]+:[0-9]+/ ]]; then
        bare="${image#*/}"
    else
        bare="$image"
    fi

    remote_name="$REGISTRY/$bare"

    echo
    echo "  Образ: $local_name"
    echo "  → $remote_name"

    if docker tag "$local_name" "$remote_name"; then
        if docker push "$remote_name"; then
            PUSHED+=("$remote_name")
            echo "  [ok] pushed"
        else
            FAILED+=("$remote_name (push failed)")
            echo "  [ОШИБКА] push завершился с ошибкой"
        fi
    else
        FAILED+=("$local_name (tag failed — образ не найден локально)")
        echo "  [ОШИБКА] tag завершился с ошибкой — образ не найден локально"
    fi
done <<< "$IMAGES"

# --- итог ---------------------------------------------------------------------
echo
echo "========================================"
echo " ИТОГ"
echo "========================================"
if [[ ${#PUSHED[@]} -gt 0 ]]; then
    echo "  Успешно запушено (${#PUSHED[@]}):"
    printf '    - %s\n' "${PUSHED[@]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  ОШИБКИ (${#FAILED[@]}):"
    printf '    - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "========================================"
echo
echo "  Чтобы использовать образы с этого registry на другом хосте, выполните:"
echo "  REGISTRY=$REGISTRY ./scripts/registry_pull.sh"
