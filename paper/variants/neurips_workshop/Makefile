.PHONY: pdf clean

pdf: main.pdf

main.pdf: main.tex references.bib neurips_2024.sty figures/hopf_phase_diagram.pdf figures/h4_detector_power_v2.pdf
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf
