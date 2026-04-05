#!/usr/bin/env bash
# =============================================================================
# archive_import.sh — импорт образов из tar.gz архивов на целевом хосте
#
# Использование:
#   ./scripts/archive_import.sh [INPUT_DIR]
#
# Примеры:
#   ./scripts/archive_import.sh
#   ./scripts/archive_import.sh /mnt/usb/images
#   INPUT_DIR=/mnt/usb/images ./scripts/archive_import.sh
#
# Переменные окружения:
#   INPUT_DIR — папка с архивами (по умолчанию: ./docker-images)
#
# Логика:
#   1. Если в INPUT_DIR есть manifest.txt — используем его (образы загружаются
#      с правильными тегами без лишних вопросов).
#   2. Иначе — загружаем все *.tar.gz файлы подряд (docker load определяет
#      теги из содержимого архива).
# =============================================================================
set -euo pipefail

INPUT_DIR="${1:-${INPUT_DIR:-./docker-images}}"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ОШИБКА: папка '$INPUT_DIR' не найдена"
    exit 1
fi

MANIFEST="$INPUT_DIR/manifest.txt"

echo "========================================"
echo " archive_import.sh"
echo "========================================"
echo "  Input dir   : $INPUT_DIR"
if [[ -f "$MANIFEST" ]]; then
    echo "  Режим        : по манифесту ($MANIFEST)"
else
    echo "  Режим        : все *.tar.gz в папке"
fi
echo "========================================"

LOADED=()
FAILED=()

load_archive() {
    local fpath="$1"
    local label="${2:-$fpath}"

    if [[ ! -f "$fpath" ]]; then
        echo "  [ОШИБКА] файл не найден: $fpath"
        FAILED+=("$label")
        return
    fi

    echo
    echo "  Загружаю: $fpath"
    if output=$(docker load < <(gzip -dc "$fpath") 2>&1); then
        echo "  [ok] $output"
        LOADED+=("$label")
    else
        echo "  [ОШИБКА] $output"
        FAILED+=("$label")
    fi
}

if [[ -f "$MANIFEST" ]]; then
    # Читаем манифест: строки вида "<image>  <filename>"
    while IFS= read -r line; do
        # Пропускаем комментарии и пустые строки
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue

        image=$(echo "$line" | awk '{print $1}')
        fname=$(echo "$line" | awk '{print $2}')
        fpath="$INPUT_DIR/$fname"

        echo
        echo "  Образ: $image"
        echo "  Файл : $fpath"

        if [[ ! -f "$fpath" ]]; then
            echo "  [ОШИБКА] файл не найден"
            FAILED+=("$image ($fname)")
            continue
        fi

        if output=$(docker load < <(gzip -dc "$fpath") 2>&1); then
            echo "  [ok] $output"
            # docker load восстанавливает тег из архива — проверяем и при
            # необходимости ретагируем (на случай если тег не совпал)
            if ! docker image inspect "$image" &>/dev/null; then
                loaded_tag=$(echo "$output" | grep -oP '(?<=Loaded image: ).*' | head -1)
                if [[ -n "$loaded_tag" && "$loaded_tag" != "$image" ]]; then
                    docker tag "$loaded_tag" "$image"
                    echo "  Ретаг: $loaded_tag → $image"
                fi
            fi
            LOADED+=("$image")
        else
            echo "  [ОШИБКА] $output"
            FAILED+=("$image")
        fi
    done < "$MANIFEST"
else
    # Нет манифеста — грузим все .tar.gz файлы
    shopt -s nullglob
    archives=("$INPUT_DIR"/*.tar.gz)
    if [[ ${#archives[@]} -eq 0 ]]; then
        echo "ОШИБКА: в '$INPUT_DIR' не найдено *.tar.gz файлов"
        exit 1
    fi

    for fpath in "${archives[@]}"; do
        load_archive "$fpath" "$(basename "$fpath")"
    done
fi

echo
echo "========================================"
echo " ИТОГ"
echo "========================================"
if [[ ${#LOADED[@]} -gt 0 ]]; then
    echo "  Загружено (${#LOADED[@]}):"
    printf '    - %s\n' "${LOADED[@]}"
fi
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  ОШИБКИ (${#FAILED[@]}):"
    printf '    - %s\n' "${FAILED[@]}"
    exit 1
fi
echo "========================================"
echo
echo "  Загруженные образы:"
docker images --format "  {{.Repository}}:{{.Tag}}  ({{.Size}})" \
    $(docker images -q | sort -u) 2>/dev/null || docker images
