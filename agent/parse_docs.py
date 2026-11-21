"""Phase 3, step 2: Docling turns each PDF into markdown, split into sections; cached as agent/parsed/<id>.json.

Run with envAgent:  python agent/parse_docs.py
The agent loop (envPytorch) only reads the JSON, so it never needs Docling.
"""
import json
import re
import time
from pathlib import Path

from docling.document_converter import DocumentConverter

HERE = Path(__file__).resolve().parent


def sections(markdown):
    """Split Docling markdown at '## ' headings -> {heading: text}; text before the first heading is 'Title'."""
    out, name, buf = {}, "Title", []
    for line in markdown.splitlines():
        m = re.match(r"^#{1,3}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                out[name] = "\n".join(buf).strip()
            name, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        out[name] = "\n".join(buf).strip()
    return out


if __name__ == "__main__":
    out_dir = HERE / "parsed"
    out_dir.mkdir(exist_ok=True)
    conv = DocumentConverter()
    for pdf in sorted((HERE / "docs").glob("*.pdf")):
        t0 = time.monotonic()
        md = conv.convert(str(pdf)).document.export_to_markdown()
        secs = sections(md)
        (out_dir / f"{pdf.stem}.json").write_text(json.dumps({"doc_id": pdf.stem, "markdown": md, "sections": secs},
                                                             indent=1), encoding="utf-8")
        print(f"{pdf.stem}: {len(secs)} sections {list(secs)} ({time.monotonic() - t0:.1f}s)", flush=True)
