MAIN = draft_v2_preprint
V1   = draft_v1

.PHONY: all v1 clean watch

all:
	latexmk -pdf $(MAIN).tex

v1:
	latexmk -pdf $(V1).tex

watch:
	latexmk -pdf -pvc $(MAIN).tex

clean:
	latexmk -C $(MAIN).tex
	latexmk -C $(V1).tex
