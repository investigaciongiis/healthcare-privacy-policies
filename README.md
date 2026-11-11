# 🔐 Criterion-Level Review of Healthcare Privacy Policies

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![GDPR](https://img.shields.io/badge/GDPR-24%20criteria-003399)
![Method](https://img.shields.io/badge/method-hybrid%20RAG-6f42c1)
![Study](https://img.shields.io/badge/study-reproducibility-2ea44f)

The experiment evaluates five publicly available Spanish healthcare privacy policies against 24 GDPR-derived criteria. A hybrid retrieval-augmented generation (RAG) pipeline combines semantic and lexical retrieval with evidence-grounded LLM assessment. Its criterion-level outputs are compared with a manually compiled study reference set.

> [!IMPORTANT]
> The generated outcomes describe **documentary coverage**. They must not be interpreted as legal determinations or certifications of GDPR compliance. The system is intended to support, rather than replace, expert human review.

## 📊 Study overview

| Component | Specification |
|---|---|
| Study sample | Five Spanish healthcare privacy policies |
| Assessment framework | 24 GDPR-derived criteria in eight thematic categories |
| Retrieval | BGE-M3 semantic retrieval, FAISS, BM25, HyDE and Reciprocal Rank Fusion |
| Assessment outputs | `SI`, `PARCIAL`, `NO_CONSTA` and `NO_APLICA` |
| Primary evaluation | Exact agreement, ordinal agreement, precision, recall and F1 |
| Evaluable comparisons | 113 policy–criterion pairs |


## 🏥 Documents included

The focused evaluation sample comprises:

- Sanitas;
- Hospital Universitario Virgen del Rocío;
- Servicio Andaluz de Salud (SAS);
- The Castilla-La Mancha health portal (SESCAM); and
- Vithas.

The documents were retrieved from their organisations' official websites. The manuscript reports the experimental retrieval date as **6 September 2026**.

## 📂 Repository structure

```text
.
├── criteria/
│   └── checklist_v1.yaml         # Spanish GDPR-derived assessment framework
├── data/
│   └── ground_truth_e1.json      # Study reference annotations
├── .env.example                  # Environment configuration template
├── compute_e1_metrics.py         # Evaluation and report generation
├── pipeline.py                   # Ingestion, segmentation, retrieval and assessment
├── requirements.txt              # Python dependencies
├── run_e1_experiment.py          # Reproducible E1 entry point
└── README.md
```

Generated policy assessments are written to `data/e1_results/`; aggregate evaluation results are written to `data/e1_metrics.json`.

## 📋 Prerequisites

- Python 3.10 or later;
- internet access for policy retrieval and initial model download;
- sufficient local storage and memory for the `BAAI/bge-m3` embedding model; and
- access to either the study's OpenAI-compatible inference infrastructure or a supported Groq endpoint.

The code is platform-independent and may be run on Linux, macOS or Windows.

## 🚀 Installation and configuration

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

Replace the placeholders with the final GitHub URL and local directory name.

### 2. Create an isolated environment

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure an inference provider

Create a local environment file from the versioned template:

Linux or macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure **one** of the following alternatives. To reproduce the
provider configuration reported in the manuscript, define all three `REMOTE_*`
variables:

```properties
REMOTE_API_URL=https://server.example/v1
REMOTE_API_KEY=replace_with_your_key
REMOTE_MODEL=gpt-oss
```

| Variable | Required | Purpose |
|---|---|---|
| `REMOTE_API_URL` | With the manuscript provider | Base URL of the OpenAI-compatible API, normally ending in `/v1` |
| `REMOTE_API_KEY` | With the manuscript provider | Authentication credential for the remote API |
| `REMOTE_MODEL` | With the manuscript provider | Model identifier exposed by the endpoint; the manuscript uses `gpt-oss` |

### Optional Hugging Face authentication

The semantic retriever uses the public `BAAI/bge-m3` embedding model from
Hugging Face. An account or access token is **not required** to download or use
this model. Unauthenticated downloads are, however, subject to lower rate
limits and may be slower. To authenticate downloads, create a Hugging Face
access token and add it to `.env`:

```properties
HF_TOKEN=replace_with_your_hugging_face_token
```

| Variable | Required | Purpose |
|---|---|---|
| `HF_TOKEN` | No | Optional authentication for Hugging Face model downloads, providing higher rate limits and potentially faster downloads |

The token is used by the Hugging Face libraries automatically. It is unrelated
to LLM inference and does not replace `REMOTE_API_KEY` or `GROQ_API_KEY`. Once
the embedding model is present in the local Hugging Face cache, subsequent runs
normally reuse the cached files.

Alternatively, the experiment supports Groq. In that case, leave the `REMOTE_*`
credentials blank and define:

```properties
GROQ_API_KEY=replace_with_your_key
```

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | With the Groq alternative | Authentication credential for the Groq API |

`python-dotenv` loads `.env` automatically when `run_e1_experiment.py` starts.
Environment variables defined by the operating system may also be used instead.

> [!CAUTION]
> Never commit API keys, access tokens or other credentials. The local `.env`
> file is excluded through `.gitignore`; `.env.example` contains placeholders
> only and is intended to remain under version control.

## 🧪 Reproducing Experiment

Run all commands from the repository root.

### 1. Assess the five policies

```bash
python run_e1_experiment.py
```

This command:

1. retrieves and cleans each privacy policy;
2. segments the document whilst preserving structural context;
3. constructs the hybrid FAISS/BM25 retriever;
4. assesses all 24 criteria and records supporting evidence; and
5. writes one JSON assessment per policy to `data/e1_results/`.

The procedure is resumable: valid results already present in the output directory are not recomputed.

### 2. Compute evaluation measures

```bash
python compute_e1_metrics.py
```

The script compares the generated assessments with `data/ground_truth_e1.json`, prints the principal measures and writes the complete report to `data/e1_metrics.json`.

Custom paths may be supplied when required:

```bash
python compute_e1_metrics.py \
  --results data/e1_results \
  --gt data/ground_truth_e1.json \
  --out data/e1_metrics.json
```

## ♻️ Reproducibility considerations

- The source policies are live web documents and may change after the study retrieval date.
- Website redesigns, redirects or anti-automation measures may affect text extraction.
- Remote model implementations may change even when the same model identifier is retained.
- Stochastic variation is reduced by setting the LLM temperature to `0.0`, but complete bit-for-bit reproducibility is not guaranteed across providers or hardware.
- `ground_truth_e1.json` is the operational study reference set, not an independently produced expert legal gold standard.

When reporting a reproduction, record the execution date, Python and dependency versions, model endpoint and identifier, and any inaccessible or modified source documents.

## 🤝 How to contribute

Contributions that improve reproducibility, documentation, extraction robustness or test coverage are welcome.

1. Fork the repository.
2. Create a focused branch:

   ```bash
   git checkout -b feature/short-description
   ```

3. Make the change and verify that the experiment scripts remain syntactically valid.
4. Do not commit credentials, generated model caches or unrelated experimental outputs.
5. Commit using a concise, descriptive message.
6. Push the branch and open a pull request.

Pull requests should explain the motivation, describe the implementation, identify any effect on the experimental protocol or outputs, and include appropriate validation evidence. Changes to the checklist, prompts, retrieval parameters, reference annotations or model configuration must be clearly identified because they may affect comparability with the manuscript.

## 🐞 Issue reporting

Please use the repository issue tracker for defects, reproducibility problems and methodological questions. A useful report should include:

- a concise title and description;
- the steps and command required to reproduce the problem;
- expected and observed behaviour;
- operating system and Python version;
- relevant dependency and model versions;
- a redacted error trace; and
- the affected policy or criterion, where applicable.

Do not include API keys, tokens or sensitive environment details.

## 📬 Contact

If you have questions or prefer to reach out by email, you can write to:  
**investigaciongiis@gmail.com**

## 📜 License

MIT

---

**Software Engineering Research Group (GIIS) · University of Murcia · Spain 🇪🇸**
