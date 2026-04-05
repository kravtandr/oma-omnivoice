#!/usr/bin/env bash
# =============================================================================
# archive_export.sh — экспорт образов из docker-compose в tar.gz архивы
#
# Использование:
#   ./scripts/archive_export.sh [COMPOSE_FILE] [OUTPUT_DIR] [--build] [--no-build]
#
# Примеры:
#   ./scripts/archive_export.sh
#   ./scripts/archive_export.sh docker-compose.yml ./images
#   OUTPUT_DIR=/mnt/usb/images ./scripts/archive_export.sh
#   BUILD=1 ./scripts/archive_export.sh
#
# Переменные окружения:
#   COMPOSE_FILE — путь к compose-файлу (по умолчанию: docker-compose.yml)
#   OUTPUT_DIR   — папка для сохранения архивов (по умолчанию: ./docker-images)
#   BUILD        — 1 = пересобрать перед экспортом (по умолчанию: 0)
#
# Создаёт:
#   OUTPUT_DIR/
#     <image-name>_<tag>.tar.gz   — по одному файлу на образ
#     manifest.txt                — список образ → файл архива
# =============================================================================
set -euo pipefail

COMPOSE_FILE="${1:-${COMPOSE_FILE:-docker-compose.yml}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-./docker-images}}"

if [[ "${3:-}" == "--build" ]]; then
    BUILD=1
elif [[ "${3:-}" == "--no-build" ]]; then
    BUILD=0
else
    BUILD="${BUILD:-0}"
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "ОШИБКА: файл '$COMPOSE_FILE' не найден"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

get_images() {
    docker compose -f "$COMPOSE_FILE" config 2>/dev/null \
        | grep -E '^\s+image:' \
        | awk '{print $2}' \
        | sort -u
}

# Имя файла из имени образа: заменяем / и : на _
image_to_filename() {
    local img="$1"
    echo "${img//\//_}" | tr ':' '_' | tr '.' '_'
}

echo "========================================"
echo " archive_export.sh"
echo "========================================"
echo "  Compose     : $COMPOSE_FILE"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Build       : $([[ $BUILD -eq 1 ]] && echo 'да' || echo 'нет')"
echo "========================================"

if [[ $BUILD -eq 1 ]]; then
    echo
    echo ">> Сборка образов..."
    docker compose -f "$COMPOSE_FILE" build
fi

echo
echo ">> Получение списка образов..."
IMAGES=$(get_images)

if [[ -z "$IMAGES" ]]; then
    echo "ОШИБКА: образы не найдены в $COMPOSE_FILE"
    exit 1
fi

echo "  Найдены образы:"
echo "$IMAGES" | sed 's/^/    - /'

# Очищаем манифест
MANIFEST="$OUTPUT_DIR/manifest.txt"
echo "# manifest: image → archive file" > "$MANIFEST"
echo "# создан: $(date -Iseconds)" >> "$MANIFEST"
echo "# compose: $COMPOSE_FILE" >> "$MANIFEST"

EXPORTED=()
FAILED=()

while IFS= read -r image; do
    [[ -z "$image" ]] && continue

    fname="$(image_to_filename "$image").tar.gz"
    fpath="$OUTPUT_DIR/$fname"

    echo
    echo "  Экспорт: $image"
    echo "  → $fpath"

    # docker save выводит tar, сжимаем налету
    if docker save "$image" | gzip -9 > "$fpath"; then
        size=$(du -sh "$fpath" | cut -f1)
        echo "  [ok] $size"
        echo "$image  $fname" >> "$MANIFEST"
        EXPORTED+=("$image → $fname ($size)")
    else
        rm -f "$fpath"
        FAILED+=("$image")
        echo "  [ОШИБКА]"
    fi
done <<< "$IMAGES"

echo
echo "========================================"
echo " ИТОГ"
echo "========================================"
if [[ ${#EXPORTED[@]} -gt 0 ]]; then
    echo "  Экспортировано (${#EXPORTED[@]}):"
    printf '    - %s\n' "${EXPORTED[@]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  ОШИБКИ (${#FAILED[@]}):"
    printf '    - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "========================================"
echo
echo "  Манифест: $MANIFEST"
echo "  Содержимое $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"
echo
echo "  Для импорта на целевом хосте:"
echo "  ./scripts/archive_import.sh $OUTPUT_DIR"
