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

paper:
	pandoc ${WRITE}/paper.md \
		--from=markdown \
		--to=pdf \
		--defaults=${WRITE}/defaults.yaml \
		--standalone \
		--bibliography=${WRITE}/papers.bib \
		--filter pandoc-citeproc \
		--output ${WRITE}/paper.pdf && \
	firefox ${WRITE}/paper.pdf

paper-html:
	pandoc ${WRITE}/paper.md \
		--from=markdown \
		--to=html \
		--defaults=${WRITE}/defaults.yaml \
		--standalone \
		--bibliography=${WRITE}/papers.bib \
		--filter pandoc-citeproc \
		--output ${WRITE}/paper.html && \
	firefox ${WRITE}/paper.html
