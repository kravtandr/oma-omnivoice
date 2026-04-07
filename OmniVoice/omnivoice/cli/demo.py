#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Gradio demo for OmniVoice.

Supports voice cloning and voice design.

Usage:
    omnivoice-demo --model /path/to/checkpoint --port 8000
"""

import argparse
import logging
import os
from typing import Any, Dict, List

_DEFAULT_NUM_STEP = max(4, min(64, int(os.environ.get("OMNIVOICE_NUM_STEP", "12"))))

import gradio as gr
import numpy as np
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
from omnivoice.utils.text import chunk_text_punctuation


def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


# ---------------------------------------------------------------------------
# Voice Design instruction templates (English UI labels)
# ---------------------------------------------------------------------------
_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": [
        "Child",
        "Teenager",
        "Young Adult",
        "Middle-aged",
        "Elderly",
    ],
    "Pitch": [
        "Very Low Pitch",
        "Low Pitch",
        "Moderate Pitch",
        "High Pitch",
        "Very High Pitch",
    ],
    "Style": ["Whisper"],
    "English Accent": [
        "American Accent",
        "Australian Accent",
        "British Accent",
        "Chinese Accent",
        "Canadian Accent",
        "Indian Accent",
        "Korean Accent",
        "Portuguese Accent",
        "Russian Accent",
        "Japanese Accent",
    ],
    "Chinese Dialect": [
        "Henan Dialect",
        "Shaanxi Dialect",
        "Sichuan Dialect",
        "Guizhou Dialect",
        "Yunnan Dialect",
        "Guilin Dialect",
        "Jinan Dialect",
        "Shijiazhuang Dialect",
        "Gansu Dialect",
        "Ningxia Dialect",
        "Qingdao Dialect",
        "Northeast Dialect",
    ],
}

_ATTR_INFO = {
    "English Accent": "Only effective for English speech.",
    "Chinese Dialect": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo",
        description="Launch a Gradio demo for OmniVoice.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="k2-fsa/OmniVoice",
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--device", default=None, help="Device to use. Auto-detected if not specified."
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP (default: 0.0.0.0).")
    parser.add_argument(
        "--port", type=int, default=7860, help="Server port (default: 7860)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy.",
    )
    parser.add_argument(
        "--share", action="store_true", default=False, help="Create public link."
    )
    return parser


# ---------------------------------------------------------------------------
# Build demo
# ---------------------------------------------------------------------------


def _split_into_sentence_groups(text: str, sentences_per_chunk: int) -> List[str]:
    """Разбивает текст на предложения, группирует по N штук."""
    sentences = chunk_text_punctuation(text.strip(), chunk_len=1)
    if not sentences:
        return [text.strip()] if text.strip() else []
    n = max(1, int(sentences_per_chunk))
    groups = []
    for i in range(0, len(sentences), n):
        group = " ".join(sentences[i : i + n])
        if group:
            groups.append(group)
    return groups


def build_demo(
    model: OmniVoice,
    checkpoint: str,
    generate_fn=None,
    voice_library=None,
) -> gr.Blocks:

    sampling_rate = model.sampling_rate

    # -- shared generation core --
    def _gen_core(
        text,
        language,
        ref_audio,
        instruct,
        num_step,
        guidance_scale,
        denoise,
        speed,
        duration,
        preprocess_prompt,
        postprocess_output,
        mode,
        ref_text=None,
        progress=None,
    ):
        def _prog(p: float, msg: str) -> None:
            if progress is not None:
                try:
                    progress(p, desc=msg)
                except Exception:
                    pass

        logging.info("Gradio: job started (mode=%s)", mode)
        _prog(0.02, "Starting…")

        if not text or not text.strip():
            return None, "Please enter the text to synthesize."

        ns = int(num_step) if num_step is not None else _DEFAULT_NUM_STEP
        ns = max(4, min(64, ns))
        gen_config = OmniVoiceGenerationConfig(
            num_step=ns,
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None

        kw: Dict[str, Any] = dict(
            text=text.strip(), language=lang, generation_config=gen_config
        )

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Please upload a reference audio."
            _prog(
                0.08,
                "Reference: load + trim (CPU). If ref_text empty: Whisper ASR next — "
                "first run downloads weights; GPU meter may stay low briefly.",
            )
            logging.info("Gradio: create_voice_clone_prompt (ASR if ref_text empty)…")
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            _prog(0.28, "Reference encoded. Running main TTS model on GPU…")

        if mode == "design":
            if instruct and instruct.strip():
                kw["instruct"] = instruct.strip()
            _prog(0.12, "Voice design: running main model on GPU…")

        if mode == "auto":
            _prog(0.12, "Auto voice: running main model on GPU…")

        try:
            logging.info("Gradio: model.generate (mode=%s) …", mode)
            audio = model.generate(**kw)
            logging.info("Gradio: generate finished (mode=%s).", mode)
        except Exception as e:
            logging.exception("Gradio: generate failed (mode=%s)", mode)
            return None, f"Error: {type(e).__name__}: {e}"

        # float32 [-1, 1] — Gradio 5 корректно пишет WAV; int16 здесь давал слабый/пустой звук в части браузеров
        waveform = audio[0].squeeze(0).numpy().astype(np.float32)
        peak = float(np.abs(waveform).max()) if waveform.size else 0.0
        if peak > 1.0:
            waveform = waveform / peak
        return (sampling_rate, waveform), "Done."

    # Allow external wrappers (e.g. spaces.GPU for ZeroGPU Spaces)
    _gen = generate_fn if generate_fn is not None else _gen_core

    # =====================================================================
    # UI
    # =====================================================================
    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
    )
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .gradio-container .prose {font-size: 1.1em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    # Reusable: language dropdown component
    def _lang_dropdown(label="Language (optional)", value="Auto"):
        return gr.Dropdown(
            label=label,
            choices=_ALL_LANGUAGES,
            value=value,
            allow_custom_value=False,
            interactive=True,
            info="Keep as Auto to auto-detect the language.",
        )

    # Reusable: optional generation settings accordion
    def _gen_settings():
        with gr.Accordion("Generation Settings (optional)", open=False):
            sp = gr.Slider(
                0.7,
                1.3,
                value=1.0,
                step=0.05,
                label="Speed",
                info="1.0 = normal. >1 faster, <1 slower. Ignored if Duration is set.",
            )
            du = gr.Number(
                value=None,
                label="Duration (seconds)",
                info=(
                    "Leave empty to use speed."
                    " Set a fixed duration to override speed."
                ),
            )
            ns = gr.Slider(
                4,
                64,
                value=_DEFAULT_NUM_STEP,
                step=1,
                label="Inference Steps",
                info=(
                    f"Default: {_DEFAULT_NUM_STEP} (env OMNIVOICE_NUM_STEP). "
                    "Lower = faster (~15s target for short text); higher = better quality."
                ),
            )
            dn = gr.Checkbox(
                label="Denoise",
                value=True,
                info="Default: enabled. Uncheck to disable denoising.",
            )
            gs = gr.Slider(
                0.0,
                4.0,
                value=2.0,
                step=0.1,
                label="Guidance Scale (CFG)",
                info="Default: 2.0.",
            )
            pp = gr.Checkbox(
                label="Preprocess Prompt",
                value=True,
                info="apply silence removal and trimming to the reference "
                "audio, add punctuation in the end of reference text (if not already)",
            )
            po = gr.Checkbox(
                label="Postprocess Output",
                value=True,
                info="Remove long silences from generated audio.",
            )
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Demo") as demo:
        gr.Markdown(
            """
# OmniVoice Demo

State-of-the-art text-to-speech model for **600+ languages**, supporting:

- **Voice Clone** — Clone any voice from a reference audio
- **Voice Design** — Create custom voices with speaker attributes
- **Auto Voice** — Let the model pick a voice; in **Text** you can use non-verbal tags (`[laughter]`, …) and pronunciation hints (see [model card](https://huggingface.co/k2-fsa/OmniVoice))

Built with [OmniVoice](https://github.com/k2-fsa/OmniVoice)
by Xiaomi Next-gen Kaldi team.
"""
        )

        with gr.Tabs():
            # ==============================================================
            # Voice Clone
            # ==============================================================
            with gr.TabItem("Voice Clone"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Text to synthesize",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Reference audio",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        gr.Markdown(
                            "<span style='font-size:0.85em;color:#888;'>"
                            "Recommended: 3–10 seconds audio. "
                            "<b>If you leave Reference text empty</b>, Whisper runs first — "
                            "the first time it downloads ~GB of weights; until then the GPU load "
                            "may look idle. Fill Reference text to skip ASR and reach synthesis faster."
                            "</span>"
                        )
                        vc_ref_text = gr.Textbox(
                            label="Reference text (optional)",
                            lines=2,
                            placeholder="Transcript of the reference audio. Leave empty"
                            " to auto-transcribe via ASR models.",
                        )
                        vc_lang = _lang_dropdown("Language (optional)")
                        (
                            vc_ns,
                            vc_gs,
                            vc_dn,
                            vc_sp,
                            vc_du,
                            vc_pp,
                            vc_po,
                        ) = _gen_settings()
                        vc_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(
                            label="Output audio",
                            type="numpy",
                        )
                        vc_status = gr.Textbox(label="Status", lines=2)

                def _clone_fn(
                    text,
                    lang,
                    ref_aud,
                    ref_text,
                    ns,
                    gs,
                    dn,
                    sp,
                    du,
                    pp,
                    po,
                    progress=gr.Progress(),
                ):
                    return _gen(
                        text,
                        lang,
                        ref_aud,
                        None,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="clone",
                        ref_text=ref_text or None,
                        progress=progress,
                    )

                vc_btn.click(
                    _clone_fn,
                    inputs=[
                        vc_text,
                        vc_lang,
                        vc_ref_audio,
                        vc_ref_text,
                        vc_ns,
                        vc_gs,
                        vc_dn,
                        vc_sp,
                        vc_du,
                        vc_pp,
                        vc_po,
                    ],
                    outputs=[vc_audio, vc_status],
                )

            # ==============================================================
            # Voice Design
            # ==============================================================
            with gr.TabItem("Voice Design"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Text to synthesize",
                            lines=4,
                            placeholder="Enter the text you want to synthesize...",
                        )
                        vd_lang = _lang_dropdown()

                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(
                                gr.Dropdown(
                                    label=_cat,
                                    choices=[_AUTO] + _choices,
                                    value=_AUTO,
                                    info=_ATTR_INFO.get(_cat),
                                )
                            )

                        (
                            vd_ns,
                            vd_gs,
                            vd_dn,
                            vd_sp,
                            vd_du,
                            vd_pp,
                            vd_po,
                        ) = _gen_settings()
                        vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(
                            label="Output audio",
                            type="numpy",
                        )
                        vd_status = gr.Textbox(label="Status", lines=2)

                def _build_instruct(groups):
                    """Extract instruct text from UI dropdowns.

                    Language unification and validation is handled by
                    _resolve_instruct inside _preprocess_all.
                    """
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    return ", ".join(selected)

                def _design_fn(
                    text, lang, ns, gs, dn, sp, du, pp, po, *groups, progress=gr.Progress()
                ):
                    return _gen(
                        text,
                        lang,
                        None,
                        _build_instruct(groups),
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="design",
                        progress=progress,
                    )

                vd_btn.click(
                    _design_fn,
                    inputs=[
                        vd_text,
                        vd_lang,
                        vd_ns,
                        vd_gs,
                        vd_dn,
                        vd_sp,
                        vd_du,
                        vd_pp,
                        vd_po,
                    ]
                    + vd_groups,
                    outputs=[vd_audio, vd_status],
                )

            # ==============================================================
            # Auto Voice
            # ==============================================================
            with gr.TabItem("Auto Voice"):
                with gr.Row():
                    with gr.Column(scale=1):
                        av_text = gr.Textbox(
                            label="Text to synthesize",
                            lines=4,
                            placeholder=(
                                "Plain text or with tags like [laughter], [sigh], "
                                "pinyin/CMU hints — see Hugging Face model card."
                            ),
                        )
                        av_lang = _lang_dropdown()
                        (
                            av_ns,
                            av_gs,
                            av_dn,
                            av_sp,
                            av_du,
                            av_pp,
                            av_po,
                        ) = _gen_settings()
                        av_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=1):
                        av_audio = gr.Audio(
                            label="Output audio",
                            type="numpy",
                        )
                        av_status = gr.Textbox(label="Status", lines=2)

                def _auto_fn(
                    text, lang, ns, gs, dn, sp, du, pp, po, progress=gr.Progress()
                ):
                    return _gen(
                        text,
                        lang,
                        None,
                        None,
                        ns,
                        gs,
                        dn,
                        sp,
                        du,
                        pp,
                        po,
                        mode="auto",
                        progress=progress,
                    )

                av_btn.click(
                    _auto_fn,
                    inputs=[
                        av_text,
                        av_lang,
                        av_ns,
                        av_gs,
                        av_dn,
                        av_sp,
                        av_du,
                        av_pp,
                        av_po,
                    ],
                    outputs=[av_audio, av_status],
                )

            # ==============================================================
            # Streaming TTS
            # ==============================================================
            with gr.TabItem("Streaming TTS"):
                gr.Markdown(
                    """
Текст разбивается на группы предложений. Аудио обновляется по мере синтеза каждой группы —
первый фрагмент появляется не дожидаясь конца всего текста.

Режим **Voice Clone**: загрузите референсное аудио. Без него используется авто-голос.
"""
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        st_text = gr.Textbox(
                            label="Text to synthesize",
                            lines=6,
                            placeholder="Введите длинный текст. Он будет синтезироваться чанками...",
                        )
                        st_spc = gr.Slider(
                            1, 10, value=1, step=1,
                            label="Sentences per chunk",
                            info="Сколько предложений объединять в один чанк синтеза.",
                        )
                        st_ref_audio = gr.Audio(
                            label="Reference audio (optional, для клонирования голоса)",
                            type="filepath",
                            elem_classes="compact-audio",
                        )
                        st_ref_text = gr.Textbox(
                            label="Reference text (optional)",
                            lines=2,
                            placeholder="Транскрипт референса. Пусто — ASR расшифрует автоматически.",
                        )
                        st_lang = _lang_dropdown()
                        (
                            st_ns,
                            st_gs,
                            st_dn,
                            st_sp,
                            st_du,
                            st_pp,
                            st_po,
                        ) = _gen_settings()
                        st_btn = gr.Button("Generate (streaming)", variant="primary")
                    with gr.Column(scale=1):
                        st_audio = gr.Audio(
                            label="Output audio (обновляется по чанкам)",
                            type="numpy",
                        )
                        st_status = gr.Textbox(label="Status", lines=3)

                def _stream_fn(
                    text,
                    sentences_per_chunk,
                    lang,
                    ref_aud,
                    ref_text,
                    ns,
                    gs,
                    dn,
                    sp,
                    du,
                    pp,
                    po,
                    progress=gr.Progress(),
                ):
                    if not text or not text.strip():
                        yield None, "Введите текст."
                        return

                    chunks = _split_into_sentence_groups(text, int(sentences_per_chunk))
                    if not chunks:
                        yield None, "Не удалось разбить текст на предложения."
                        return

                    total = len(chunks)
                    progress(0, desc=f"Разбито на {total} чанков")

                    ns_i = max(4, min(64, int(ns) if ns is not None else _DEFAULT_NUM_STEP))
                    gen_config = OmniVoiceGenerationConfig(
                        num_step=ns_i,
                        guidance_scale=float(gs) if gs is not None else 2.0,
                        denoise=bool(dn) if dn is not None else True,
                        preprocess_prompt=bool(pp),
                        postprocess_output=bool(po),
                    )
                    lang_val = lang if (lang and lang != "Auto") else None

                    # Создаём clone-prompt один раз (содержит ASR, если ref_text пустой)
                    voice_prompt = None
                    if ref_aud:
                        progress(0.05, desc="Обработка референса (ASR если нет ref_text)…")
                        try:
                            voice_prompt = model.create_voice_clone_prompt(
                                ref_audio=ref_aud,
                                ref_text=ref_text or None,
                            )
                        except Exception as e:
                            logging.exception("Streaming: create_voice_clone_prompt failed")
                            yield None, f"Ошибка референса: {e}"
                            return

                    accumulated = None
                    sr = model.sampling_rate

                    for i, chunk_text in enumerate(chunks):
                        frac = (i + 1) / total
                        progress(frac, desc=f"Чанк {i + 1}/{total}: {chunk_text[:50]}…")

                        kw: Dict[str, Any] = dict(
                            text=chunk_text,
                            language=lang_val,
                            generation_config=gen_config,
                        )
                        if sp is not None and float(sp) != 1.0:
                            kw["speed"] = float(sp)
                        if du is not None and float(du) > 0:
                            kw["duration"] = float(du)
                        if voice_prompt is not None:
                            kw["voice_clone_prompt"] = voice_prompt
                        elif not ref_aud:
                            pass  # auto voice

                        try:
                            audio = model.generate(**kw)
                        except Exception as e:
                            logging.exception("Streaming: chunk %d failed", i + 1)
                            yield (
                                (sr, accumulated) if accumulated is not None else None,
                                f"Чанк {i + 1}/{total} — ОШИБКА: {e}",
                            )
                            continue

                        chunk_arr = audio[0].squeeze(0).numpy().astype(np.float32)
                        peak = float(np.abs(chunk_arr).max()) if chunk_arr.size else 0.0
                        if peak > 1.0:
                            chunk_arr = chunk_arr / peak

                        accumulated = (
                            chunk_arr if accumulated is None
                            else np.concatenate([accumulated, chunk_arr])
                        )
                        yield (sr, accumulated), f"Чанк {i + 1}/{total} готов."

                    yield (sr, accumulated) if accumulated is not None else None, f"Готово ({total} чанков)."

                st_btn.click(
                    _stream_fn,
                    inputs=[
                        st_text,
                        st_spc,
                        st_lang,
                        st_ref_audio,
                        st_ref_text,
                        st_ns,
                        st_gs,
                        st_dn,
                        st_sp,
                        st_du,
                        st_pp,
                        st_po,
                    ],
                    outputs=[st_audio, st_status],
                )

            # ==============================================================
            # Voice Library
            # ==============================================================
            with gr.TabItem("Voice Library"):
                if voice_library is None:
                    gr.Markdown(
                        "**Библиотека голосов недоступна.** "
                        "Запустите сервис через API с переменной `OMNIVOICE_VOICES_DIR`."
                    )
                else:
                    def _vl_table_data():
                        rows = []
                        for m in voice_library.list_all():
                            rows.append([
                                m["name"],
                                m.get("ref_text", "") or "",
                                m.get("added_at", ""),
                                f"{m.get('size_bytes', 0) // 1024} KB",
                            ])
                        return rows or [["—", "—", "—", "—"]]

                    def _vl_names():
                        return voice_library.names()

                    gr.Markdown("""
### Библиотека референсных голосов

Загружайте аудио (3–20 с), давайте имя — и используйте голос по имени без повторной загрузки.
""")
                    with gr.Row():
                        # ── Left column: управление библиотекой ──────────────────
                        with gr.Column(scale=1):
                            with gr.Accordion("Добавить голос", open=True):
                                vl_name = gr.Textbox(
                                    label="Имя голоса",
                                    placeholder="Например: Диктор Алиса",
                                )
                                vl_audio = gr.Audio(
                                    label="Аудио референса",
                                    type="filepath",
                                    elem_classes="compact-audio",
                                )
                                vl_ref_text = gr.Textbox(
                                    label="Транскрипт референса (необязательно)",
                                    lines=2,
                                    placeholder="Текст, произнесённый в референсном аудио. "
                                                "Оставьте пустым — Whisper расшифрует автоматически.",
                                )
                                vl_add_btn = gr.Button("Добавить в библиотеку", variant="primary")
                            vl_add_status = gr.Textbox(label="Статус", lines=2, interactive=False)

                            gr.Markdown("---")
                            vl_refresh_btn = gr.Button("Обновить список")
                            vl_table = gr.Dataframe(
                                headers=["Имя", "Транскрипт", "Добавлен", "Размер"],
                                value=_vl_table_data(),
                                interactive=False,
                                wrap=True,
                            )
                            with gr.Row():
                                vl_del_name = gr.Dropdown(
                                    label="Удалить голос",
                                    choices=_vl_names(),
                                    value=None,
                                    allow_custom_value=False,
                                )
                                vl_del_btn = gr.Button("Удалить", variant="stop")
                            vl_del_status = gr.Textbox(label="Статус удаления", lines=1, interactive=False)

                        # ── Right column: синтез с голосом из библиотеки ─────────
                        with gr.Column(scale=1):
                            gr.Markdown("#### Синтез с голосом из библиотеки")
                            vl_synth_voice = gr.Dropdown(
                                label="Голос",
                                choices=_vl_names(),
                                value=None,
                                allow_custom_value=False,
                                info="Выберите голос из библиотеки.",
                            )
                            vl_preview_audio = gr.Audio(
                                label="Превью голоса",
                                type="filepath",
                                interactive=False,
                                elem_classes="compact-audio",
                            )
                            vl_synth_text = gr.Textbox(
                                label="Текст для синтеза",
                                lines=4,
                                placeholder="Введите текст...",
                            )
                            vl_synth_lang = _lang_dropdown("Язык (необязательно)")
                            (
                                vl_ns, vl_gs, vl_dn, vl_sp, vl_du, vl_pp, vl_po
                            ) = _gen_settings()
                            vl_synth_btn = gr.Button("Синтезировать", variant="primary")
                            vl_synth_audio = gr.Audio(label="Результат", type="numpy")
                            vl_synth_status = gr.Textbox(label="Статус", lines=2, interactive=False)

                    # ── Callbacks ────────────────────────────────────────────────

                    def _all_updates():
                        names = _vl_names()
                        data = _vl_table_data()
                        return data, gr.update(choices=names, value=None), gr.update(choices=names, value=None)

                    def _vl_add_fn(name, audio_path, ref_text):
                        if not name or not name.strip():
                            return ("Введите имя голоса.",) + _all_updates()
                        if not audio_path:
                            return ("Загрузите аудио-файл.",) + _all_updates()
                        try:
                            with open(audio_path, "rb") as f:
                                audio_bytes = f.read()
                            voice_library.add(
                                name=name.strip(),
                                audio_bytes=audio_bytes,
                                ref_text=ref_text or None,
                            )
                            return (f"Голос {name.strip()!r} добавлен.",) + _all_updates()
                        except ValueError as e:
                            return (f"Ошибка: {e}",) + _all_updates()
                        except Exception as e:
                            logging.exception("VoiceLibrary add failed")
                            return (f"Ошибка: {e}",) + _all_updates()

                    def _vl_delete_fn(name):
                        if not name:
                            return ("Выберите голос для удаления.",) + _all_updates()
                        ok = voice_library.delete(name)
                        msg = f"Голос {name!r} удалён." if ok else f"Голос {name!r} не найден."
                        return (msg,) + _all_updates()

                    def _vl_refresh_fn():
                        return _all_updates()

                    def _vl_preview_fn(voice_name):
                        if not voice_name:
                            return None
                        return voice_library.get_audio_path(voice_name)

                    def _vl_synth_fn(
                        voice_name, text, lang, ns, gs, dn, sp, du, pp, po,
                        progress=gr.Progress(),
                    ):
                        if not voice_name:
                            return None, "Выберите голос из библиотеки."
                        audio_path = voice_library.get_audio_path(voice_name)
                        if audio_path is None:
                            return None, f"Голос {voice_name!r} не найден (файл отсутствует)."
                        ref_text_val = voice_library.get_ref_text(voice_name)
                        return _gen(
                            text, lang, audio_path, None,
                            ns, gs, dn, sp, du, pp, po,
                            mode="clone",
                            ref_text=ref_text_val,
                            progress=progress,
                        )

                    vl_synth_voice.change(
                        _vl_preview_fn,
                        inputs=[vl_synth_voice],
                        outputs=[vl_preview_audio],
                    )
                    vl_add_btn.click(
                        _vl_add_fn,
                        inputs=[vl_name, vl_audio, vl_ref_text],
                        outputs=[vl_add_status, vl_table, vl_del_name, vl_synth_voice],
                    )
                    vl_del_btn.click(
                        _vl_delete_fn,
                        inputs=[vl_del_name],
                        outputs=[vl_del_status, vl_table, vl_del_name, vl_synth_voice],
                    )
                    vl_refresh_btn.click(
                        _vl_refresh_fn,
                        inputs=[],
                        outputs=[vl_table, vl_del_name, vl_synth_voice],
                    )
                    vl_synth_btn.click(
                        _vl_synth_fn,
                        inputs=[
                            vl_synth_voice, vl_synth_text, vl_synth_lang,
                            vl_ns, vl_gs, vl_dn, vl_sp, vl_du, vl_pp, vl_po,
                        ],
                        outputs=[vl_synth_audio, vl_synth_status],
                    )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or get_best_device()

    checkpoint = args.model
    if not checkpoint:
        parser.print_help()
        return 0
    logging.info(f"Loading model from {checkpoint}, device={device} ...")
    model = OmniVoice.from_pretrained(
        checkpoint,
        device_map=device,
        dtype=torch.float16,
        load_asr=True,
    )
    print("Model loaded.")

    demo = build_demo(model, checkpoint)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
