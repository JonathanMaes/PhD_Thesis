#TEXFILES = $(wildcard *.tex)
FILES = phd.tex
PDFS = $(FILES:.tex=.pdf)
OUT = build

TEX = pdflatex -output-directory=$(OUT)

all: $(PDFS)

$(PDFS): %.pdf: %.tex
	$(TEX) $(@:.pdf=.tex)
	cp -r bib $(OUT)
	cd $(OUT); bibtex $(@:.pdf=)
	$(TEX) $(@:.pdf=.tex)
	makeindex $(OUT)/$(@:.pdf=.idx)
	$(TEX) $(@:.pdf=.tex)

.PHONY: clean
clean:
	rm -f *.aux  *.blg *.ind *.ilg *.log *.toc *.out *.idx *.bbl #$(PDFS) 
