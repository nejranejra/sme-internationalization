# SME R&D Intensity and Firm Performance

Research project for *ExInt II: Research Designs in SME Research* (WU Vienna, SS 2026).

Reproducible pipeline: pull from WRDS Compustat Global, clean a firm-year panel,
build variables and figures, estimate fixed-effects regressions, and render a
research note to PDF.

## Research Question
Does R&D intensity positively affect firm performance among European SMEs, and
does firm size moderate this relationship?

## Hypotheses
- H1: R&D intensity affects RoA (expensing effect expected in the short run).
- H2: Firm size positively moderates the R&D intensity–RoA relationship.

> Note on DOI: `pifo` (foreign income) is not available in Compustat Global, and
> `rect/sale` produces extreme outliers for SMEs. The research question was
> updated to R&D intensity, which has full coverage.

## Variables

| Variable       | Field(s)    | Formula              | Role           |
|----------------|-------------|----------------------|----------------|
| RoA            | ib, at      | ib / at              | Dependent (Y)  |
| R&D intensity  | xrd, at     | xrd.fillna(0) / at   | Independent (X)|
| R&D x Size     | -           | rd_intensity x ln_at | H2 interaction |
| Firm size      | at          | log(at)              | Moderator+Ctrl |
| Leverage       | dltt, at    | dltt / at            | Control        |
| CAPX intensity | capx, at    | capx / at            | Control        |
| Cash ratio     | che, at     | che / at             | Control        |

## Data

| Item         | Detail                            |
|--------------|-----------------------------------|
| Source       | WRDS / Compustat Global           |
| Table        | comp_global_daily.g_funda         |
| Downloaded   | [FILL IN today's date]            |
| License      | WRDS subscriber agreement         |
| Fiscal years | 2015-2024                         |
| Raw rows     | [FILL IN from pull_metadata.txt]  |
| Clean rows   | [FILL IN from clean_log.txt]      |

## How to reproduce

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
# then edit .env and set WRDS_USERNAME
task all
```

`task all` runs: pull → clean → descriptives → regression → render.

## Project structure

```
code/01_pull_data.py      pull raw Compustat Global from WRDS
code/02_clean.py          build clean firm-year panel
code/03_descriptives.py   variables, summary stats, figures
code/04_regression.py     three regression models + diagnostics
research_note.qmd          Quarto source of the research note
references/library.bib     bibliography
references/apa.csl         APA citation style
Taskfile.yml               pipeline definition
```

Repository: https://github.com/nejranejra/sme-internationalization
