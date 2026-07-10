.PHONY: pdf clean

pdf: main.pdf

main.pdf: main.tex references.bib figures/centered_hopf_validation.pdf
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf
