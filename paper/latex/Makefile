TEX = diff_mppi
LATEXMK = latexmk
PDFLATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean distclean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib
	$(PDFLATEX) $(TEX)
	$(BIBTEX) $(TEX)
	$(PDFLATEX) $(TEX)
	$(PDFLATEX) $(TEX)

latexmk: $(TEX).tex references.bib
	$(LATEXMK) -pdf -pdflatex="pdflatex -interaction=nonstopmode" $(TEX)

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out
	rm -f $(TEX).fdb_latexmk $(TEX).fls $(TEX).synctex.gz
	rm -f $(TEX).toc $(TEX).lof $(TEX).lot $(TEX).nav $(TEX).snm

distclean: clean
	rm -f $(TEX).pdf
