ROOT:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
WRITE:=${ROOT}/write

deps:
	pip install -Ur requirements_dev.txt

lint:
	isort --check-only **/*.py
	black --check **/*.py
	flake8 .

format:
	black **/*.py
	isort **/*.py

open-paper:
	firefox ${WRITE}/paper.pdf

paper-no-open:
	pandoc ${WRITE}/paper.md \
		--from=markdown \
		--to=pdf \
		--defaults=${WRITE}/defaults.yaml \
		--standalone \
		--bibliography=${WRITE}/papers.bib \
		--filter pandoc-citeproc \
		--output ${WRITE}/paper.pdf

paper: paper-no-open open-paper

paper-html:
	pandoc ${WRITE}/paper.md \
		--from=markdown \
		--to=html \
		--defaults=${WRITE}/defaults.yaml \
		--standalone \
		--bibliography=${WRITE}/papers.bib \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--output ${WRITE}/paper.html && \
	firefox ${WRITE}/paper.html

paper-with-cover: paper-no-open
	gs -dNOPAUSE -sDEVICE=pdfwrite -sOUTPUTFILE=write/bw_msu_ms_writing_proj_nov_2020.pdf -dBATCH write/coverpage.pdf write/paper.pdf
	firefox ${WRITE}/bw_msu_ms_writing_proj_nov_2020.pdf
