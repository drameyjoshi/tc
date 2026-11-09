MAIN = asnotes
SOURCES = $(wildcard *.tex) $(wildcard *.bib)

all: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -c

distclean:
	latexmk -C
