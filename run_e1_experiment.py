"""
run_e1_experiment.py — Manuscript Experiment E1: five Spanish healthcare privacy policies.

Corpus:
  Sanitas, Hospital Virgen del Rocío, SAS Andalucía,
  Sanidad Castilla-La Mancha and Vithas.

Results are written to data/e1_results/.
The procedure is resumable: previously audited policies are skipped automatically,
including any results available in the legacy partial-batch directory.

Usage:
  python run_e1_experiment.py
"""
import os, sys, json, time, shutil, traceback
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from pipeline import (
    load_policy, chunk_policy_hybrid,
    build_retriever, run_audit, load_checklist,
    build_remote_provider, LLMPool, LLMProvider,
)

CHECKLIST_PATH    = Path("criteria/checklist_v1.yaml")
RESULTS_DIR       = Path("data/e1_results")
LEGACY_HEALTH_DIR = Path("data/e1_salud_results")   # Historical on-disk path retained for compatibility.

SECTOR_LABELS = {
    "healthcare": "healthcare",
    "public_healthcare": "public healthcare",
}

POLICIES = [
    # ── Healthcare ────────────────────────────────────────────────────────────────
    {
        "url":    "https://www.sanitas.es/seguros/politica-privacidad",
        "sector": "healthcare",
        "name":   "Sanitas",
        "language": "es",
    },
    {
        "url":    "https://hospitaluvrocio.es/politica-de-privacidad/",
        "sector": "healthcare",
        "name":   "Hospital Virgen del Rocio",
        "language": "es",
    },
    {
        "url":    "https://www.sspa.juntadeandalucia.es/servicioandaluzdesalud/politica-de-privacidad",
        "sector": "public_healthcare",
        "name":   "SAS Andalucia",
        "language": "es",
    },
    {
        "url":    "https://sanidad.castillalamancha.es/content/politica-de-privacidad",
        "sector": "public_healthcare",
        "name":   "Sanidad Castilla-La Mancha",
        "language": "es",
    },
    {
        "url":    "https://www.vithas.es/politica-de-privacidad",
        "sector": "healthcare",
        "name":   "Vithas",
        "language": "es",
    }
]


def build_llm_pool() -> LLMPool:
    providers = []

    remote_url = os.getenv("REMOTE_API_URL", "").strip()
    remote_key = os.getenv("REMOTE_API_KEY", "").strip()
    if remote_url and remote_key:
        prov = build_remote_provider(
            base_url=remote_url,
            api_key=remote_key,
            model=os.getenv("REMOTE_MODEL", "gpt-oss"),
        )
        if prov:
            providers.append(prov)
            print(f"  [LLM] Remote server -> {prov.name}")

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            llm_groq = ChatGroq(
                api_key=groq_key,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
            )
            providers.append(LLMProvider(
                name="meta-llama/llama-4-scout-17b-16e-instruct",
                llm=llm_groq,
                max_calls=900,
            ))
            print("  [LLM] Groq enabled")
        except Exception as e:
            print(f"  [LLM] Groq unavailable: {e}")

    if not providers:
        sys.exit("ERROR: Set REMOTE_API_KEY or GROQ_API_KEY in the environment or .env file")

    return LLMPool(providers, call_delay=12.0)


def _load_completed_from_dir(results_dir: Path) -> dict[str, Path]:
    """Return a mapping from source URL to JSON path for every audit in ``results_dir``."""
    completed = {}
    for p in results_dir.glob("*_audit.json"):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            url = d.get("document", {}).get("doc_source", "")
            if url:
                completed[url] = p
        except Exception:
            pass
    return completed


def load_completed(results_dir: Path) -> set[str]:
    """
    Return the set of URLs that have already been processed.

    Search both ``results_dir`` and the legacy partial-batch directory so that results from a
    previous partial run can be reused. Files found in the legacy directory are
    copied automatically to ``results_dir``.
    """
    completed = _load_completed_from_dir(results_dir)

    if LEGACY_HEALTH_DIR.exists():
        legacy = _load_completed_from_dir(LEGACY_HEALTH_DIR)
        for url, src_path in legacy.items():
            if url not in completed:
                dst = results_dir / src_path.name
                if not dst.exists():
                    shutil.copy2(src_path, dst)
                    print(f"  [Resume] Copied from legacy partial-batch results: {src_path.name}")
                completed[url] = dst

    return set(completed.keys())


def save_result(policy_meta: dict, audit: dict, results_dir: Path) -> Path:
    name_slug = policy_meta["name"].lower().replace(" ", "_")
    out = results_dir / f"{name_slug}_audit.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    return out


def print_summary(policies: list) -> None:
    from collections import Counter
    counts = Counter(p["sector"] for p in policies)
    print("  Sectors:")
    for sector, n in sorted(counts.items()):
        print(f"    {SECTOR_LABELS.get(sector, sector):<22} {n} policy/policies")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    completed = load_completed(RESULTS_DIR)
    pending   = [p for p in POLICIES if p["url"] not in completed]

    print(f"\n{'='*70}")
    print(f"E1 — HEALTHCARE PRIVACY-POLICY ASSESSMENT  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checklist : {CHECKLIST_PATH}")
    print(f"Results   : {RESULTS_DIR}")
    print(f"Total     : {len(POLICIES)}  |  Completed: {len(completed)}  |  Pending: {len(pending)}")
    print(f"{'='*70}")
    print_summary(POLICIES)
    print(f"{'='*70}\n")

    if not pending:
        print("All policies have already been assessed.")
        print(f"Results: {RESULTS_DIR}")
        return

    llm = build_llm_pool()
    checklist = load_checklist(CHECKLIST_PATH)
    n_crit = len(checklist["criteria"])
    print(f"\n  [Checklist] {n_crit} criteria loaded from {CHECKLIST_PATH.name}\n")

    errors  = []
    t_start = time.time()

    for idx, pol in enumerate(pending, start=1):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{idx}/{len(pending)}] {ts}  {pol['name']}  ({SECTOR_LABELS.get(pol['sector'], pol['sector'])})")
        print(f"  URL: {pol['url']}")
        print("-" * 70)

        try:
            doc = load_policy(pol["url"])
            print(f"  [1/4] Loaded   : {len(doc.clean_text):,} characters ({doc.source_type})")

            if len(doc.clean_text) < 500:
                raise ValueError(
                    f"Policy text is too short ({len(doc.clean_text)} characters); "
                    "the page may have failed to load or may block automated access"
                )

            chunks = chunk_policy_hybrid(doc)
            print(f"  [2/4] Chunks   : {len(chunks)}")

            retrieve = build_retriever(chunks)
            print(f"  [3/4] FAISS+BM25 index ready")

            print(f"  [4/4] Assessing {n_crit} GDPR-derived criteria...")
            audit = run_audit(
                chunks      = chunks,
                retrieve_fn = retrieve,
                llm         = llm,
                checklist   = checklist,
                sector      = pol["sector"],
                language    = pol["language"],
                verbose     = True,
            )

            out = save_result(pol, audit, RESULTS_DIR)
            print(f"  OK  Saved: {out.name}  (score={audit['total_score_pct']}%)\n")

        except KeyboardInterrupt:
            print("\n[Manual interruption — progress has been saved]")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            errors.append({"name": pol["name"], "url": pol["url"], "error": str(e)})
            print()

    elapsed  = time.time() - t_start
    done     = len(pending) - len(errors)
    total_ok = len(completed) + done

    print(f"\n{'='*70}")
    print(f"Current session -> Completed: {done}  |  Errors: {len(errors)}")
    print(f"Cumulative total: {total_ok}/{len(POLICIES)}  |  Session duration: {elapsed/60:.1f} min")
    if errors:
        print("\nPolicies with errors:")
        for e in errors:
            print(f"  {e['name']}: {e['error']}")
    print(f"\nResults: {RESULTS_DIR}")
    print("Next step: review the reference annotations and run compute_e1_metrics.py")


if __name__ == "__main__":
    main()
