.DEFAULT_GOAL := train
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= $(PROJECT_ROOT)/.venv/bin/python
ARGS ?=

.PHONY: train test typecheck
train:
	"$(PYTHON)" "$(PROJECT_ROOT)/train.py" $(ARGS)

test:
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m pytest -q

typecheck:
	cd "$(PROJECT_ROOT)" && "$(PYTHON)" -m mypy
