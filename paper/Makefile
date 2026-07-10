.PHONY: pdf clean watch arxiv

pdf: main.pdf

main.pdf: main.tex references.bib figures/centered_hopf_validation.pdf figures/centered_hopf_robustness.pdf
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf

watch:
	latexmk -pvc -pdf main.tex

# Build the arXiv-ready source tarball at dist/arxiv_v0.4.0.tar.gz.
# Delegates to scripts/arxiv_build.sh — see that script for the full
# pipeline (figure discovery, pdflatex/bibtex, .bbl shipping, tar packing,
# simulated extract+rebuild sanity check, page-count verification).
arxiv:
	cd .. && bash scripts/arxiv_build.sh
