TEX ?= contact_rich_diff_mppi
LATEXMK = latexmk
PDFLATEX = pdflatex
BIBTEX = bibtex

.PHONY: all figures legacy systems clean distclean

all: figures $(TEX).pdf

figures:
	python3 ../../scripts/render_contact_submission_figures.py \
		--output-dir ../figures/submission

legacy:
	$(MAKE) TEX=diff_mppi diff_mppi.pdf

systems:
	$(MAKE) TEX=cudanav_systems cudanav_systems.pdf

$(TEX).pdf: $(TEX).tex references.bib
	$(PDFLATEX) $(TEX)
	$(BIBTEX) $(TEX)
	$(PDFLATEX) $(TEX)
	$(PDFLATEX) $(TEX)

latexmk: $(TEX).tex references.bib
	$(LATEXMK) -pdf -pdflatex="pdflatex -interaction=nonstopmode" $(TEX)

clean:
	rm -f contact_rich_diff_mppi.aux contact_rich_diff_mppi.bbl
	rm -f contact_rich_diff_mppi.blg contact_rich_diff_mppi.log
	rm -f contact_rich_diff_mppi.out contact_rich_diff_mppi.pdf
	rm -f contact_rich_diff_mppi.fdb_latexmk contact_rich_diff_mppi.fls
	rm -f contact_rich_diff_mppi.synctex.gz
	rm -f diff_mppi.aux diff_mppi.bbl diff_mppi.blg diff_mppi.log
	rm -f diff_mppi.out diff_mppi.fdb_latexmk diff_mppi.fls
	rm -f diff_mppi.synctex.gz
	rm -f cudanav_systems.aux cudanav_systems.bbl
	rm -f cudanav_systems.blg cudanav_systems.log
	rm -f cudanav_systems.out cudanav_systems.pdf
	rm -f cudanav_systems.fdb_latexmk cudanav_systems.fls
	rm -f cudanav_systems.synctex.gz
	rm -f *.toc *.lof *.lot *.nav *.snm

distclean: clean
	rm -f diff_mppi.pdf
