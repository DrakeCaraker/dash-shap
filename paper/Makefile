# Auto-detect drafts: find all draft_v*.tex, set MAIN to highest version number
ALL_DRAFTS  := $(basename $(wildcard draft_v*.tex))
MAIN        := $(shell ls draft_v*.tex 2>/dev/null | sed 's/.*_v\([0-9]*\).*/\1 &/' | sort -n | tail -1 | awk '{print $$2}' | sed 's/\.tex$$//')
DRAFTS      := $(filter-out $(MAIN),$(ALL_DRAFTS))
LATEXMK     = latexmk
LATEX_FLAGS = -pdf

.PHONY: all draft clean watch diff wordcount lint

all:
	$(LATEXMK) $(LATEX_FLAGS) $(MAIN).tex

draft:
	$(LATEXMK) $(LATEX_FLAGS) $(V).tex

watch:
	$(LATEXMK) $(LATEX_FLAGS) -pvc $(MAIN).tex

diff:
	latexdiff $(V).tex $(MAIN).tex > diff.tex
	$(LATEXMK) $(LATEX_FLAGS) diff.tex

wordcount:
	@texcount -inc -total $(MAIN).tex

lint:
	@chktex -q $(MAIN).tex || true

clean:
	$(LATEXMK) -c $(MAIN).tex $(foreach d,$(DRAFTS),$(d).tex)
	rm -f diff.tex diff.pdf
