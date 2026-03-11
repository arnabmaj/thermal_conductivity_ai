"""
extract_kappa.py — Extract thermal conductivity data from PDFs using Claude API.

Usage:
    python pipelines/extract_kappa.py data/raw/papers/*.pdf

Outputs (saved to data/extracted/ by default):
    results.csv            — flat, validated records
    results.json           — full raw extraction per PDF
    rejected_records.json  — records that failed validation (for debugging)
"""

import anthropic
import base64
import json
import csv
import sys
import os
from pathlib import Path

from dotenv import load_dotenv


# ── Config ───────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DOTENV_CANDIDATES = (
    _PROJECT_ROOT / "config" / "secrets.env",
    _PROJECT_ROOT / "config" / "@config" / "secrets.env",
)

# If a key is already set in the environment, keep it (override=False).
# This only wires configuration; extraction logic is unchanged.
for _p in _DOTENV_CANDIDATES:
    if _p.exists():
        load_dotenv(_p, override=False)
        break

# Default to the lowest-cost model available to this project/key.
# You can override via ANTHROPIC_MODEL in `config/secrets.env` or your shell env.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = 4000
MAX_TOKENS_RETRY = 8000
DEFAULT_OUT_DIR = "data/extracted"

SYSTEM_PROMPT = """You are a materials science expert extracting thermal conductivity data
from research papers. Your only job is to return clean, structured JSON — nothing else.
No markdown fences, no preamble, no commentary. Raw JSON only."""

EXTRACTION_PROMPT = """Extract only thermal conductivity data from this paper.

Return ONLY a JSON object in this exact format:

{
  "paper_title": "string or null",
  "has_thermal_conductivity_data": true,
  "records": [
    {
      "material": "compound/material name exactly as written",
      "thermal_conductivity_w_mk": 0.0,
      "temperature_k": 0.0,
      "pressure_gpa": 0.0,
      "crystal_structure": "structure name or null",
      "space_group": "space group or null",
      "kappa_type": "lattice|total|electronic|unknown",
      "direction": "direction if anisotropic, else null",
      "method": "experimental|bte|md|dft|unknown",
      "condition": "brief condition description or null",
      "page": 0,
      "evidence": "short verbatim supporting snippet",
      "confidence": "high|medium|low"
      "source_type": "text|table|figure|unknown",
    }
  ]
}

Rules:
- Extract ONLY thermal conductivity values.
- Only extract space group or crystal structure if explicitly stated in the text.Do not infer them.
- Do NOT extract: superconducting Tc, Curie temperature, Neel temperature,
  synthesis/annealing/melting/decomposition temperatures, or any temperature
  unrelated to thermal conductivity.
- thermal_conductivity_w_mk must be in W/mK (W m^-1 K^-1). If the paper uses
  another notation for the same unit, normalize it to W/mK. If not enough
  information exists to convert safely, skip that record.
- temperature_k can be null if not reported explicitly.
- pressure_gpa can be null if not reported explicitly.
- crystal_structure and space_group can be null if not reported.
- Return one record per material per condition. If the same material is reported
  at different temperatures, pressures, directions, methods, or phases,
  return separate records.
- If the paper reports lattice thermal conductivity specifically, set kappa_type="lattice".
  If total thermal conductivity, set kappa_type="total". If electronic only, set
  kappa_type="electronic". Otherwise use "unknown".
- direction should capture anisotropic labels like a-axis, c-axis, [100], etc.
- method should be one of: experimental, bte, md, dft, unknown.
- evidence must be a short verbatim snippet (under 25 words) from the paper.
- If no thermal conductivity data exists, return has_thermal_conductivity_data=false
  and records=[].
- Return raw JSON only. Do not truncate the records array.
"""


# ── Validation ───────────────────────────────────────────────────────────────

# Terms that strongly suggest the value is NOT thermal conductivity
DISQUALIFYING = {
    "superconduct",
    "critical temperature",
    "curie",
    "neel",
    "transition temperature",
    "melting",
    "annealing",
    "synthesis temperature",
    "decomposition temperature",
    "tc(",
    " t_c",
}

# Evidence should ideally contain at least one of these
QUALIFYING = {
    "thermal conductivity",
    "lattice thermal",
    "heat transport",
    "phonon transport",
    "w m",
    "w/m",
    "wk",
    "w m-1 k-1",
    "w m^-1 k^-1",
}

VALID_KAPPA_TYPES = {"lattice", "total", "electronic", "unknown"}
VALID_METHODS = {"experimental", "bte", "md", "dft", "unknown"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_record(record: dict) -> tuple[bool, str]:
    material = (record.get("material") or "").strip()
    evidence = (record.get("evidence") or "").lower().strip()
    kappa = record.get("thermal_conductivity_w_mk")
    temperature = record.get("temperature_k")
    pressure = record.get("pressure_gpa")
    kappa_type = (record.get("kappa_type") or "unknown").lower().strip()
    method = (record.get("method") or "unknown").lower().strip()
    confidence = (record.get("confidence") or "").lower().strip()
    page = record.get("page")

    if not material:
        return False, "material is empty"

    if kappa is None:
        return False, "thermal_conductivity_w_mk is null"
    if not _is_number(kappa) or kappa <= 0:
        return False, f"thermal_conductivity_w_mk is not a positive number: {kappa!r}"

    if temperature is not None and (not _is_number(temperature) or temperature <= 0):
        return False, f"temperature_k is invalid: {temperature!r}"

    if pressure is not None and (not _is_number(pressure) or pressure < 0):
        return False, f"pressure_gpa is invalid: {pressure!r}"

    if page is None or not isinstance(page, int) or page < 0:
        return False, f"page is invalid: {page!r}"

    if kappa_type not in VALID_KAPPA_TYPES:
        return False, f"invalid kappa_type: {kappa_type!r}"

    if method not in VALID_METHODS:
        return False, f"invalid method: {method!r}"

    if confidence not in VALID_CONFIDENCE:
        return False, f"invalid confidence: {confidence!r}"

    if any(term in evidence for term in DISQUALIFYING):
        return False, f"evidence suggests unrelated property: '{evidence[:120]}'"

    if evidence and not any(term in evidence for term in QUALIFYING):
        # Allow records through if field names themselves strongly indicate thermal conductivity context
        if kappa_type == "unknown" and method == "unknown":
            return False, f"evidence lacks thermal-conductivity signal: '{evidence[:120]}'"

    return True, ""


# ── API call ─────────────────────────────────────────────────────────────────

def pdf_to_base64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def call_api(client: anthropic.Anthropic, pdf_data: str, max_tokens: int) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ],
    )
    raw = response.content[0].text.strip()

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]

    return raw.strip()


def extract_from_pdf(client: anthropic.Anthropic, pdf_path: str) -> dict:
    print(f"  Processing: {Path(pdf_path).name}")
    pdf_data = pdf_to_base64(pdf_path)

    raw = call_api(client, pdf_data, MAX_TOKENS)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as first_err:
        print(f"    JSON truncated or invalid — retrying with {MAX_TOKENS_RETRY} tokens...")
        raw = call_api(client, pdf_data, MAX_TOKENS_RETRY)
        try:
            result = json.loads(raw)
            print("    Retry succeeded.")
        except json.JSONDecodeError:
            raise ValueError(
                f"JSON parse failed even after retry.\n"
                f"First error: {first_err}\n"
                f"Raw tail: ...{raw[-300:]}"
            )

    result["source_file"] = Path(pdf_path).name
    return result


# ── Pipeline ─────────────────────────────────────────────────────────────────

def process_pdfs(pdf_paths: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    client = anthropic.Anthropic()
    all_results: list[dict] = []
    valid_rows: list[dict] = []
    rejected_records: list[dict] = []

    print(f"\nProcessing {len(pdf_paths)} PDF(s)...\n")

    for path in pdf_paths:
        try:
            result = extract_from_pdf(client, path)
            all_results.append(result)

            if not result.get("has_thermal_conductivity_data"):
                print("  -> No thermal conductivity data\n")
                continue

            n_valid = 0
            for record in result.get("records", []):
                is_valid, reason = validate_record(record)
                flat = {
                    "source_file": result["source_file"],
                    "paper_title": result.get("paper_title", ""),
                    **record,
                }

                if is_valid:
                    valid_rows.append(flat)
                    n_valid += 1
                else:
                    rejected_records.append({**flat, "rejection_reason": reason})
                    print(f"    REJECTED: {record.get('material', '?')} — {reason}")

            n_total = len(result.get("records", []))
            n_rejected = n_total - n_valid
            status = "OK" if n_rejected == 0 else "OK (with rejections)"
            print(f"  {status}  {n_valid} valid  |  {n_rejected} rejected\n")

        except Exception as e:
            print(f"  FAIL  {Path(path).name}: {e}\n")
            all_results.append({
                "source_file": Path(path).name,
                "error": str(e),
                "records": [],
            })

    return all_results, valid_rows, rejected_records


# ── Output ───────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "source_file",
    "paper_title",
    "material",
    "thermal_conductivity_w_mk",
    "temperature_k",
    "pressure_gpa",
    "crystal_structure",
    "space_group",
    "kappa_type",
    "direction",
    "method",
    "condition",
    "page",
    "confidence",
    "evidence"
]


def save_results(
    all_results: list[dict],
    valid_rows: list[dict],
    rejected_records: list[dict],
    out_dir: str = DEFAULT_OUT_DIR,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    if valid_rows:
        with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid_rows)
        print(f"Saved: {out / 'results.csv'}  ({len(valid_rows)} records)")
    else:
        print("No valid records — results.csv not written.")

    if rejected_records:
        with open(out / "rejected_records.json", "w", encoding="utf-8") as f:
            json.dump(rejected_records, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out / 'rejected_records.json'}  ({len(rejected_records)} rejected)")

    # Use ASCII to avoid Windows console encoding issues.
    print("\n-- Summary ------------------------------------------------")
    for doc in all_results:
        print(f"\n{doc['source_file']}")
        if "error" in doc:
            print(f"  ERROR: {doc['error']}")
            continue
        if not doc.get("has_thermal_conductivity_data"):
            print("  (no thermal conductivity data)")
            continue

        for r in doc.get("records", []):
            kappa = r.get("thermal_conductivity_w_mk")
            kappa_str = f"{kappa} W/mK" if kappa is not None else "N/A"
            temp = r.get("temperature_k")
            temp_str = f", T={temp} K" if temp is not None else ""
            press = r.get("pressure_gpa")
            press_str = f", P={press} GPa" if press is not None else ""
            direction = f", dir={r.get('direction')}" if r.get("direction") else ""
            method = r.get("method", "")
            conf = r.get("confidence", "")
            print(f"  {r.get('material', 'UNKNOWN')}: {kappa_str}{temp_str}{press_str}{direction}  ({method}, {conf})")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipelines/extract_kappa.py data/raw/papers/paper1.pdf ...")
        sys.exit(1)

    pdf_files = [p for p in sys.argv[1:] if p.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in arguments.")
        sys.exit(1)

    all_results, valid_rows, rejected = process_pdfs(pdf_files)
    save_results(all_results, valid_rows, rejected, out_dir=DEFAULT_OUT_DIR)