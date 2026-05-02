.PHONY: pdf clean watch

pdf: main.pdf

main.pdf: main.tex references.bib figures/hopf_phase_diagram.pdf figures/h4_detector_power.pdf figures/ell1_phase_boundary.pdf
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf

watch:
	latexmk -pvc -pdf main.tex
