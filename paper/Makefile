.PHONY: pdf clean watch arxiv

pdf: main.pdf

main.pdf: main.tex references.bib figures/hopf_phase_diagram.pdf figures/h4_detector_power_v2.pdf figures/ell1_phase_boundary.pdf figures/lambda_scaling_loglog.pdf figures/limit_cycle_supercritical.pdf figures/codim2_phase_diagram.pdf figures/hawkes_sv_equivalence.pdf figures/mckean_vlasov_propagation_chaos.pdf figures/kappa_star_robustness_heatmap.pdf figures/kappa_star_misspecification_curve.pdf figures/h1_synthetic_ordering.pdf figures/stationary_density_2d_kde.pdf figures/sw2_sample_complexity.pdf figures/ell1_closed_form.tex
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.pdf

watch:
	latexmk -pvc -pdf main.tex

# Build the arXiv-ready source tarball at dist/arxiv_v0.3.3.tar.gz.
# Delegates to scripts/arxiv_build.sh — see that script for the full
# pipeline (figure discovery, pdflatex/bibtex, .bbl shipping, tar packing,
# simulated extract+rebuild sanity check, page-count verification).
arxiv:
	cd .. && bash scripts/arxiv_build.sh
