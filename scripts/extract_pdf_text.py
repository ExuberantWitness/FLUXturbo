"""Extract text from ARIS PDFs for ccchain testing."""
import os
from PyPDF2 import PdfReader

PDF_DIR = r"E:\DATA\vscode\ARIS\pdf"
OUT_DIR = r"E:\DATA\vscode\FLUXturbo\scripts\pdf_texts"
os.makedirs(OUT_DIR, exist_ok=True)

for fname in sorted(os.listdir(PDF_DIR)):
    if not fname.endswith(".pdf"):
        continue
    path = os.path.join(PDF_DIR, fname)
    reader = PdfReader(path)
    full_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            full_text.append(t)

    all_text = "\n\n".join(full_text)
    out_path = os.path.join(OUT_DIR, fname.replace(".pdf", ".txt"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"{fname}: {len(reader.pages)} pages, {len(all_text)} chars -> {out_path}")
    # Print first 500 chars as preview
    print(all_text[:500])
    print("...\n")
