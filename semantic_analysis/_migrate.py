"""
Script to migrate semantic_analysis M1-M6 into Criminal_detection/semantic_analysis/
with updated module names and import paths.
"""
import shutil
from pathlib import Path

SRC = Path('d:/Projects/Semantic_analysis')
DST = Path('d:/Projects/Criminal_detection/semantic_analysis')


def copy_rewrite(src_path: Path, dst_path: Path, replacements: list[tuple[str, str]]) -> None:
    txt = src_path.read_text(encoding='utf-8')
    for old, new in replacements:
        txt = txt.replace(old, new)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(txt, encoding='utf-8')
    print(f'  Wrote: {dst_path.relative_to(DST)}')


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + '\n', encoding='utf-8')
    print(f'  Wrote: {path.relative_to(DST)}')


# ─── video/ (was M1) ───────────────────────────────────────────────────────────
V1_REPL = [
    ('from .config import M1Config, DEFAULT_CONFIG', 'from .config import VideoConfig, DEFAULT_CONFIG'),
    ('class M1Config:', 'class VideoConfig:'),
    ('M1Config', 'VideoConfig'),
    ('from M3.sampler import sample_frames as m3_sample_frames',
     'from semantic_analysis.ocr.sampler import sample_frames as m3_sample_frames'),
    ('from M3.config import M3Config',
     'from semantic_analysis.ocr.config import OcrConfig as M3Config'),
    ('M3Config(', 'OcrConfig('),
]

for fname in ['config.py', 'sampler.py', 'embedder.py', 'aggregator.py', 'validator.py', 'writer.py', 'pipeline.py']:
    src = SRC / 'M1' / fname
    if src.exists():
        copy_rewrite(src, DST / 'video' / fname, V1_REPL)

write(DST / 'video' / '__init__.py', '''
# video/ — Video/VLM module (was M1)
from .config import VideoConfig, DEFAULT_CONFIG
from .embedder import CLIPEmbedder
from .pipeline import process_video
from .validator import validate_segment_records

__all__ = [
    "VideoConfig",
    "DEFAULT_CONFIG",
    "CLIPEmbedder",
    "process_video",
    "validate_segment_records",
]
''')

# ─── audio/ (was M2) ───────────────────────────────────────────────────────────
A2_REPL = [
    ('from M4.config import DEFAULT_CONFIG as m4_cfg',
     'from semantic_analysis.llm_rag.retrieval.config import DEFAULT_CONFIG as m4_cfg'),
]

src = SRC / 'M2' / 'pipeline.py'
copy_rewrite(src, DST / 'audio' / 'pipeline.py', A2_REPL)

write(DST / 'audio' / '__init__.py', '''
# audio/ — Audio/NLP module (was M2)
from .pipeline import process_video_audio

__all__ = ["process_video_audio"]
''')

# ─── ocr/ (was M3) ─────────────────────────────────────────────────────────────
OCR_REPL = [
    ('from .config import M3Config', 'from .config import OcrConfig'),
    ('class M3Config:', 'class OcrConfig:'),
    ('M3Config', 'OcrConfig'),
    # YOLO model path — point to centralized models dir
    ('"yolov8n.pt"', 'str(Path(__file__).resolve().parent.parent / "models" / "yolo" / "yolov8n.pt")'),
    ("'yolov8n.pt'", 'str(Path(__file__).resolve().parent.parent / "models" / "yolo" / "yolov8n.pt")'),
]

for fname in ['config.py', 'sampler.py', 'detector.py', 'aggregator.py', 'validator.py', 'writer.py', 'pipeline.py']:
    src = SRC / 'M3' / fname
    if src.exists():
        copy_rewrite(src, DST / 'ocr' / fname, OCR_REPL)

write(DST / 'ocr' / '__init__.py', '''
# ocr/ — Vision Intelligence module (was M3)
from .config import OcrConfig, DEFAULT_CONFIG
from .pipeline import process_video

__all__ = ["OcrConfig", "DEFAULT_CONFIG", "process_video"]
''')

# ─── llm_rag/retrieval/ (was M4) ────────────────────────────────────────────────
M4_REPL = [
    ('from .config import M4Config, DEFAULT_CONFIG', 'from .config import RetrievalConfig as M4Config, DEFAULT_CONFIG'),
    ('class M4Config:', 'class RetrievalConfig:'),
    ('M4Config', 'RetrievalConfig'),
]

for fname in ['config.py', 'index_manager.py', 'joiner.py', 'scorer.py', 'search.py', 'validator.py', 'pipeline.py']:
    src = SRC / 'M4' / fname
    if src.exists():
        copy_rewrite(src, DST / 'llm_rag' / 'retrieval' / fname, M4_REPL)

write(DST / 'llm_rag' / 'retrieval' / '__init__.py', '''
# llm_rag/retrieval/ — Retrieval module (was M4)
from .config import RetrievalConfig, DEFAULT_CONFIG
from .pipeline import run_ingestion
from .search import search

__all__ = ["RetrievalConfig", "DEFAULT_CONFIG", "run_ingestion", "search"]
''')

# ─── llm_rag/ (was M5) ─────────────────────────────────────────────────────────
# NOTE: query_parser.py and generator.py are replaced with LOCAL versions
# Only copy schemas.py, retrieval_stub.py, _retry.py
M5_SRC = SRC / 'M5-llm-rag'

M5_REPL = [
    ('from M4.config import DEFAULT_CONFIG as m4_cfg',
     'from semantic_analysis.llm_rag.retrieval.config import DEFAULT_CONFIG as m4_cfg'),
    ('from M4.search import search as m4_search',
     'from semantic_analysis.llm_rag.retrieval.search import search as m4_search'),
]

for fname in ['schemas.py', 'retrieval_stub.py', '_retry.py']:
    src = M5_SRC / fname
    if src.exists():
        copy_rewrite(src, DST / 'llm_rag' / fname, M5_REPL)

write(DST / 'llm_rag' / '__init__.py', '''
# llm_rag/ — Local query parsing + template answer generation (was M5)
# NOTE: No cloud API required. Works fully offline.
from .schemas import StructuredQuery, AttributedObject, M5Answer, CitedSegment
from .query_parser import parse_query
from .generator import generate_answer
from .pipeline import run_pipeline, format_answer_for_display

__all__ = [
    "StructuredQuery", "AttributedObject", "M5Answer", "CitedSegment",
    "parse_query",
    "generate_answer",
    "run_pipeline",
    "format_answer_for_display",
]
''')

print('\nAll module files migrated!')
