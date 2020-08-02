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
		--output ${WRITE}/paper.pdf && \
	firefox ${WRITE}/paper.pdf
