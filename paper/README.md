# Paper (ICML 2025 LaTeX)

## Files

- `paper.tex`: main manuscript
- `references.bib`: BibTeX database

## Build

From this folder:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

If `pdflatex` is not installed, install a TeX distribution (e.g., TeX Live).
