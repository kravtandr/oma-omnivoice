"""HTTP API for OmniVoice TTS (voice clone, design, auto; text supports non-verbal tags & pronunciation hints)."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import struct
import tempfile
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, Literal, List, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger(__name__)

# Меньше шагов → быстрее синтез (качество чуть ниже). Рекомендация OmniVoice: 16 для скорости.
_DEFAULT_NUM_STEP = max(4, min(64, int(os.environ.get("OMNIVOICE_NUM_STEP", "12"))))

_model = None
_device: str = "cuda"
_model_id: str = "k2-fsa/OmniVoice"


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off", ""):
        return False
    raise ValueError(f"invalid boolean: {v!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device, _model_id
    from omnivoice import OmniVoice

    _model_id = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
    _device = os.environ.get("OMNIVOICE_DEVICE") or _best_device()
    load_asr = os.environ.get("OMNIVOICE_PRELOAD_ASR", "").lower() in (
        "1",
        "true",
        "yes",
    )
    asr_model_name = os.environ.get("OMNIVOICE_ASR_MODEL", "openai/whisper-large-v3-turbo")

    logger.info("Loading OmniVoice %s on %s (preload_asr=%s)...", _model_id, _device, load_asr)
    _model = OmniVoice.from_pretrained(
        _model_id,
        device_map=_device,
        dtype=torch.float16,
        load_asr=load_asr,
        asr_model_name=asr_model_name,
    )
    logger.info(
        "Model ready, sampling_rate=%s, default num_step=%s (OMNIVOICE_NUM_STEP)",
        _model.sampling_rate,
        _DEFAULT_NUM_STEP,
    )

    if not getattr(app.state, "omnivoice_gradio_mounted", False):
        import gradio as gr

        from omnivoice.cli.demo import build_demo

        gradio_path = os.environ.get("OMNIVOICE_GRADIO_PATH", "/gradio").rstrip("/") or "/gradio"
        if not gradio_path.startswith("/"):
            gradio_path = "/" + gradio_path
        root_path = os.environ.get("OMNIVOICE_GRADIO_ROOT_PATH") or None
        demo_blocks = build_demo(_model, _model_id)
        # default_concurrency_limit=1: не допускаем параллельные generate() на одной GPU (OOM/зависание).
        gr.mount_gradio_app(
            app,
            demo_blocks.queue(default_concurrency_limit=1, max_size=8),
            path=gradio_path,
            root_path=root_path,
        )
        # mount_gradio_app подменяет lifespan FastAPI чтобы запустить queue при старте приложения.
        # Но если вызывается внутри уже запущенного lifespan — подмена игнорируется,
        # queue никогда не стартует, задачи вечно висят в ожидании.
        # Запускаем queue вручную:
        demo_blocks.run_startup_events()
        await demo_blocks.run_extra_startup_events()
        app.state.omnivoice_gradio_mounted = True
        app.state.omnivoice_gradio_path = gradio_path
        app.state.omnivoice_demo_blocks = demo_blocks
        logger.info("Gradio UI mounted at %s", gradio_path)

    yield
    _model = None
    demo = getattr(app.state, "omnivoice_demo_blocks", None)
    if demo is not None:
        demo._queue.close()


app = FastAPI(
    title="OmniVoice TTS API",
    version="1.0.0",
    lifespan=lifespan,
    description="""
## Режимы

- **Voice Cloning** — загрузите `ref_audio` (WAV и т.д.); опционально `ref_text` (если нет — Whisper расшифрует).
- **Voice Design** — поле `instruct`, например `female, low pitch, british accent`.
- **Auto Voice** — без `ref_audio` и без `instruct`.

## Non-verbal и произношение

Всё задаётся в **`text`**: теги вроде `[laughter]`, китайская транскрипция, английские CMU-фонемы в скобках — как в
[карточке модели](https://huggingface.co/k2-fsa/OmniVoice).

Интерактивный **Gradio** доступен по пути `/gradio` (или значение переменной `OMNIVOICE_GRADIO_PATH`).

**Скорость:** по умолчанию `num_step` = переменная `OMNIVOICE_NUM_STEP` (если не задана — 12) для укладывания в ~15 с на короткую фразу на типичной GPU.
Длинный текст, chunking, первый запуск Whisper без `ref_text` могут занять заметно дольше.
""",
)


class GenerateJsonBody(BaseModel):
    """JSON-вариант запроса; эталонное аудио — base64 (или используйте multipart `/v1/generate`)."""

    text: str = Field(..., description="Текст синтеза, включая теги [laughter] и подсказки произношения.")
    ref_audio_base64: Optional[str] = Field(
        None, description="Опционально: WAV/другой формат в base64 для клонирования голоса."
    )
    ref_text: Optional[str] = None
    instruct: Optional[str] = None
    language: Optional[str] = None
    num_step: int = Field(
        default=_DEFAULT_NUM_STEP,
        description="Шаги диффузии; меньше — быстрее.",
    )
    guidance_scale: float = 2.0
    speed: float = 1.0
    duration: Optional[float] = None
    t_shift: float = 0.1
    denoise: bool = True
    postprocess_output: bool = True
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    response_format: Literal["wav", "json"] = "wav"


@app.get("/", include_in_schema=False)
def root():
    path = getattr(app.state, "omnivoice_gradio_path", "/gradio")
    return RedirectResponse(path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": _device,
        "gradio_path": getattr(app.state, "omnivoice_gradio_path", None),
    }


@app.get("/manifest.json", include_in_schema=False)
def web_app_manifest():
    """Браузер запрашивает /manifest.json для PWA; без маршрута сыпятся 404 в логах."""
    gradio_path = getattr(app.state, "omnivoice_gradio_path", "/gradio")
    return JSONResponse(
        {
            "name": "OmniVoice TTS",
            "short_name": "OmniVoice",
            "start_url": gradio_path,
            "display": "browser",
        },
        media_type="application/manifest+json",
    )


def _run_generate(
    *,
    text: str,
    ref_path: Optional[str],
    ref_text: Optional[str],
    instruct: Optional[str],
    language: Optional[str],
    num_step: int,
    guidance_scale: float,
    speed: float,
    duration: Optional[float],
    t_shift: float,
    denoise: bool,
    postprocess_output: bool,
    layer_penalty_factor: float,
    position_temperature: float,
    class_temperature: float,
):
    assert _model is not None
    audios = _model.generate(
        text=text,
        language=language,
        ref_audio=ref_path,
        ref_text=ref_text,
        instruct=instruct,
        duration=duration,
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed,
        t_shift=t_shift,
        denoise=denoise,
        postprocess_output=postprocess_output,
        layer_penalty_factor=layer_penalty_factor,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
    )
    return audios[0], _model.sampling_rate


def _tensor_to_wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
    """PCM16 WAV: float16/тихий сигнал после torchaudio.save часто превращался в «немой» файл."""
    w = waveform.detach().cpu().float().contiguous()
    if w.dim() == 2:
        if w.size(0) == 1:
            arr = w.squeeze(0).numpy()
        else:
            arr = w.mean(dim=0).numpy()
    elif w.dim() == 1:
        arr = w.numpy()
    else:
        arr = w.reshape(-1).numpy()

    peak = float(np.abs(arr).max()) if arr.size else 0.0
    logger.info(
        "WAV encode: samples=%s peak=%.5f sr=%s dtype_was=%s",
        arr.size,
        peak,
        sample_rate,
        waveform.dtype,
    )
    if peak < 1e-8:
        logger.warning("WAV encode: сигнал по сути нулевой")
    elif peak < 0.08:
        arr = (arr / peak) * 0.99
    else:
        arr = np.clip(arr, -1.0, 1.0)

    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _streaming_wav_header(sample_rate: int) -> bytes:
    """WAV-заголовок для стриминга: размеры выставлены в 0xFFFFFFFF (unknown length)."""
    channels, bits = 1, 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate,
        byte_rate, block_align, bits,
        b"data", 0xFFFFFFFF,
    )


def _tensor_to_pcm16(waveform: torch.Tensor) -> bytes:
    """Только сырые PCM16 байты без WAV-заголовка."""
    w = waveform.detach().cpu().float().contiguous()
    if w.dim() == 2:
        arr = w.squeeze(0).numpy() if w.size(0) == 1 else w.mean(dim=0).numpy()
    elif w.dim() == 1:
        arr = w.numpy()
    else:
        arr = w.reshape(-1).numpy()
    peak = float(np.abs(arr).max()) if arr.size else 0.0
    if peak < 1e-8:
        pass
    elif peak < 0.08:
        arr = (arr / peak) * 0.99
    else:
        arr = np.clip(arr, -1.0, 1.0)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _split_into_sentence_groups(text: str, sentences_per_chunk: int) -> List[str]:
    """Разбивает текст на предложения, затем группирует по N штук."""
    from omnivoice.utils.text import chunk_text_punctuation
    # chunk_len=1 → минимальный размер чанка → каждое предложение отдельно
    sentences = chunk_text_punctuation(text.strip(), chunk_len=1)
    if not sentences:
        return [text.strip()] if text.strip() else []
    n = max(1, sentences_per_chunk)
    groups = []
    for i in range(0, len(sentences), n):
        group = " ".join(sentences[i : i + n])
        if group:
            groups.append(group)
    return groups


@app.post(
    "/v1/generate",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "application/json": {},
            }
        }
    },
)
async def generate_multipart(
    text: Annotated[str, Form(..., description="Текст; поддерживаются [laughter] и подсказки произношения.")],
    ref_audio: Optional[UploadFile] = File(default=None),
    ref_text: Annotated[Optional[str], Form()] = None,
    instruct: Annotated[Optional[str], Form()] = None,
    language: Annotated[Optional[str], Form()] = None,
    num_step: Annotated[int, Form()] = _DEFAULT_NUM_STEP,
    guidance_scale: Annotated[float, Form()] = 2.0,
    speed: Annotated[float, Form()] = 1.0,
    duration: Annotated[Optional[float], Form()] = None,
    t_shift: Annotated[float, Form()] = 0.1,
    denoise: Annotated[str, Form()] = "true",
    postprocess_output: Annotated[str, Form()] = "true",
    layer_penalty_factor: Annotated[float, Form()] = 5.0,
    position_temperature: Annotated[float, Form()] = 5.0,
    class_temperature: Annotated[float, Form()] = 0.0,
    response_format: Annotated[Literal["wav", "json"], Form()] = "wav",
):
    """Синтез через multipart: для клонирования прикрепите файл `ref_audio`."""
    tmp_path: Optional[str] = None
    try:
        try:
            d_b = _str2bool(denoise)
            p_b = _str2bool(postprocess_output)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

        if ref_audio is not None:
            raw = await ref_audio.read()
        else:
            raw = b""
        if raw:
            name = (ref_audio.filename or "ref") if ref_audio is not None else "ref"
            suffix = os.path.splitext(name)[1] or ".wav"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(raw)
            ref_for_model = tmp_path
        else:
            ref_for_model = None

        wav, sr = await asyncio.to_thread(
            _run_generate,
            text=text,
            ref_path=ref_for_model,
            ref_text=ref_text,
            instruct=instruct,
            language=language,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            duration=duration,
            t_shift=t_shift,
            denoise=d_b,
            postprocess_output=p_b,
            layer_penalty_factor=layer_penalty_factor,
            position_temperature=position_temperature,
            class_temperature=class_temperature,
        )

        if response_format == "json":
            raw_wav = _tensor_to_wav_bytes(wav, sr)
            b64 = base64.standard_b64encode(raw_wav).decode("ascii")
            return JSONResponse(
                {
                    "sample_rate": sr,
                    "audio_wav_base64": b64,
                    "format": "wav",
                }
            )

        return Response(
            content=_tensor_to_wav_bytes(wav, sr),
            media_type="audio/wav",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("generate failed")
        raise HTTPException(500, detail=str(e)) from e
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post("/v1/generate/json")
def generate_json(body: GenerateJsonBody):
    """Синтез из JSON; эталонное аудио передаётся как base64."""
    tmp_path: Optional[str] = None
    try:
        if body.ref_audio_base64:
            raw = base64.standard_b64decode(body.ref_audio_base64)
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(raw)
            ref_for_model = tmp_path
        else:
            ref_for_model = None

        wav, sr = _run_generate(
            text=body.text,
            ref_path=ref_for_model,
            ref_text=body.ref_text,
            instruct=body.instruct,
            language=body.language,
            num_step=body.num_step,
            guidance_scale=body.guidance_scale,
            speed=body.speed,
            duration=body.duration,
            t_shift=body.t_shift,
            denoise=body.denoise,
            postprocess_output=body.postprocess_output,
            layer_penalty_factor=body.layer_penalty_factor,
            position_temperature=body.position_temperature,
            class_temperature=body.class_temperature,
        )

        if body.response_format == "json":
            raw_wav = _tensor_to_wav_bytes(wav, sr)
            b64 = base64.standard_b64encode(raw_wav).decode("ascii")
            return {
                "sample_rate": sr,
                "audio_wav_base64": b64,
                "format": "wav",
            }

        return Response(
            content=_tensor_to_wav_bytes(wav, sr),
            media_type="audio/wav",
        )
    except Exception as e:
        logger.exception("generate_json failed")
        raise HTTPException(500, detail=str(e)) from e
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@app.post(
    "/v1/generate/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": (
                "Streaming WAV (заголовок с size=0xFFFFFFFF, затем PCM16 чанки).\n\n"
                "Текст разбивается на группы предложений (`sentences_per_chunk`), "
                "каждая группа синтезируется отдельно — первый чанк отдаётся сразу, "
                "не дожидаясь конца всего текста."
            ),
        }
    },
)
async def generate_stream(
    text: Annotated[str, Form(..., description="Текст синтеза.")],
    sentences_per_chunk: Annotated[int, Form(ge=1, le=20)] = 1,
    ref_audio: Optional[UploadFile] = File(default=None),
    ref_text: Annotated[Optional[str], Form()] = None,
    instruct: Annotated[Optional[str], Form()] = None,
    language: Annotated[Optional[str], Form()] = None,
    num_step: Annotated[int, Form()] = _DEFAULT_NUM_STEP,
    guidance_scale: Annotated[float, Form()] = 2.0,
    speed: Annotated[float, Form()] = 1.0,
    duration: Annotated[Optional[float], Form()] = None,
    t_shift: Annotated[float, Form()] = 0.1,
    denoise: Annotated[str, Form()] = "true",
    postprocess_output: Annotated[str, Form()] = "true",
    layer_penalty_factor: Annotated[float, Form()] = 5.0,
    position_temperature: Annotated[float, Form()] = 5.0,
    class_temperature: Annotated[float, Form()] = 0.0,
):
    """Потоковый синтез: аудио отдаётся по мере готовности каждой группы предложений.

    Формат ответа: streaming WAV (заголовок + сырые PCM16-чанки).
    Совместим с ffplay, VLC, ffmpeg и большинством аудио-библиотек.

    Пример воспроизведения:
    ```
    curl -s -X POST http://localhost:8080/v1/generate/stream \\
      -F "text=Первое предложение. Второе предложение. Третье." \\
      -F "sentences_per_chunk=1" | ffplay -i pipe:0
    ```
    """
    tmp_path: Optional[str] = None
    try:
        try:
            d_b = _str2bool(denoise)
            p_b = _str2bool(postprocess_output)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

        chunks = _split_into_sentence_groups(text, sentences_per_chunk)
        if not chunks:
            raise HTTPException(400, "Текст пустой или не удалось разбить на предложения.")

        logger.info(
            "stream: %d чанков (sentences_per_chunk=%d) из %d символов",
            len(chunks), sentences_per_chunk, len(text),
        )

        # Сохраняем ref_audio один раз — используется для всех чанков
        if ref_audio is not None:
            raw = await ref_audio.read()
        else:
            raw = b""
        if raw:
            name = (ref_audio.filename or "ref") if ref_audio is not None else "ref"
            suffix = os.path.splitext(name)[1] or ".wav"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(raw)
            ref_for_model = tmp_path
        else:
            ref_for_model = None

        sample_rate = _model.sampling_rate  # type: ignore[union-attr]

        async def _stream_generator() -> AsyncIterator[bytes]:
            yield _streaming_wav_header(sample_rate)
            for i, chunk_text in enumerate(chunks):
                logger.info("stream: чанк %d/%d: %r", i + 1, len(chunks), chunk_text[:60])
                try:
                    wav, _ = await asyncio.to_thread(
                        _run_generate,
                        text=chunk_text,
                        ref_path=ref_for_model,
                        ref_text=ref_text,
                        instruct=instruct,
                        language=language,
                        num_step=num_step,
                        guidance_scale=guidance_scale,
                        speed=speed,
                        duration=duration,
                        t_shift=t_shift,
                        denoise=d_b,
                        postprocess_output=p_b,
                        layer_penalty_factor=layer_penalty_factor,
                        position_temperature=position_temperature,
                        class_temperature=class_temperature,
                    )
                    yield _tensor_to_pcm16(wav)
                except Exception:
                    logger.exception("stream: ошибка в чанке %d", i + 1)
                    # Продолжаем со следующим чанком, не обрываем поток
                finally:
                    pass

        return StreamingResponse(
            _stream_generator(),
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(sample_rate),
                "X-Chunks-Total": str(len(chunks)),
                "Transfer-Encoding": "chunked",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("generate_stream failed")
        raise HTTPException(500, detail=str(e)) from e
    # tmp_path убирается после завершения стриминга через background task не нужен —
    # файл живёт пока генератор не отработает; для упрощения оставляем его до перезапуска
    # (tmpdir OS очистит при ребуте; для продакшена добавить weakref/finalizer)
