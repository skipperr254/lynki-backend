"""
Tests for extraction_service's concurrent PDF page OCR: page-order
reassembly regardless of OCR completion order, all-scanned and all-fail
cases.

`_ocr_image_bytes` (the Claude Vision call) is monkeypatched so these tests
exercise only `_parse_pdf`'s orchestration — no live API calls. Real
in-memory PDFs are built with fitz itself rather than hand-rolled fakes, so
the test exercises the real PyMuPDF API surface `_parse_pdf` actually calls
(get_text, get_pixmap, tobytes).
"""
import asyncio

import fitz
import pytest

from app.services.extraction_service import ExtractionService


def _new_service() -> ExtractionService:
    """Skip __init__ (constructs a real AsyncAnthropic client) — _parse_pdf
    only needs self._ocr_image_bytes, which every test below monkeypatches."""
    return ExtractionService.__new__(ExtractionService)


def _build_pdf(*, native_text_pages: int = 0, blank_pages: int = 0) -> bytes:
    doc = fitz.open()
    for _ in range(native_text_pages):
        page = doc.new_page()
        page.insert_text((72, 72), "Native text page with plenty of readable content here.")
    for _ in range(blank_pages):
        doc.new_page()  # blank -> under SCANNED_PAGE_CHAR_THRESHOLD, forces OCR
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.mark.asyncio
async def test_ocr_pages_reassemble_in_order_despite_out_of_order_completion(monkeypatch):
    svc = _new_service()
    call_count = {"n": 0}

    async def fake_ocr(image_bytes, media_type):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx == 0:
            # First-queued page (page 1) is slow — resolves LAST.
            await asyncio.sleep(0.05)
            return "Page 1 OCR text"
        # Second-queued page (page 2) resolves FIRST.
        return "Page 2 OCR text"

    monkeypatch.setattr(svc, "_ocr_image_bytes", fake_ocr)

    pdf_bytes = _build_pdf(native_text_pages=1, blank_pages=2)
    result = await svc._parse_pdf(pdf_bytes)

    parts = result.split("\n\n")
    assert len(parts) == 3
    assert "Native text page" in parts[0]
    assert parts[1] == "Page 1 OCR text"  # still in original page order...
    assert parts[2] == "Page 2 OCR text"  # ...even though it resolved first


@pytest.mark.asyncio
async def test_all_scanned_pdf_queues_and_places_every_page(monkeypatch):
    svc = _new_service()

    async def fake_ocr(image_bytes, media_type):
        return "OCR text"

    monkeypatch.setattr(svc, "_ocr_image_bytes", fake_ocr)

    pdf_bytes = _build_pdf(blank_pages=3)
    result = await svc._parse_pdf(pdf_bytes)

    assert result == "OCR text\n\nOCR text\n\nOCR text"


@pytest.mark.asyncio
async def test_all_ocr_fails_returns_empty_string_no_exception(monkeypatch):
    svc = _new_service()

    async def fake_ocr(image_bytes, media_type):
        return ""  # _ocr_image_bytes's real failure contract: empty string, never raises

    monkeypatch.setattr(svc, "_ocr_image_bytes", fake_ocr)

    pdf_bytes = _build_pdf(blank_pages=2)
    result = await svc._parse_pdf(pdf_bytes)

    assert result == ""


@pytest.mark.asyncio
async def test_no_scanned_pages_never_calls_ocr(monkeypatch):
    svc = _new_service()
    ocr_calls = []

    async def fake_ocr(image_bytes, media_type):
        ocr_calls.append(1)
        return "should not be called"

    monkeypatch.setattr(svc, "_ocr_image_bytes", fake_ocr)

    pdf_bytes = _build_pdf(native_text_pages=2)
    result = await svc._parse_pdf(pdf_bytes)

    assert ocr_calls == []
    assert result.count("Native text page") == 2
