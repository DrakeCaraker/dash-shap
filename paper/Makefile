MAIN        = draft_v3_preprint
V2          = draft_v2_preprint
V1          = draft_v1
DRAFTS      = $(MAIN) $(V2) $(V1)
LATEXMK     = latexmk
LATEX_FLAGS = -pdf

.PHONY: all v1 clean watch diff wordcount lint

all:
	$(LATEXMK) $(LATEX_FLAGS) $(MAIN).tex

v1:
	$(LATEXMK) $(LATEX_FLAGS) $(V1).tex

watch:
	$(LATEXMK) $(LATEX_FLAGS) -pvc $(MAIN).tex

diff:
	latexdiff $(V1).tex $(MAIN).tex > diff.tex
	$(LATEXMK) $(LATEX_FLAGS) diff.tex

wordcount:
	@texcount -inc -total $(MAIN).tex

lint:
	@chktex -q $(MAIN).tex || true

clean:
	$(LATEXMK) -c $(foreach d,$(DRAFTS),$(d).tex)
	rm -f diff.tex diff.pdf
