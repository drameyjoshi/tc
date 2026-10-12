MAIN = asnotes
SOURCES = $(MAIN).tex c1.tex c2.tex c2p.tex $(MAIN).bib

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c

distclean:
	latexmk -C
