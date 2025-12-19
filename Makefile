# ================================================================
#  - Installs PyTorch + CVXPY stack
#  - Choose: make setup-cpu   or   make setup-cu124
# ================================================================

SHELL	 := /bin/bash
VENV_DIR ?= $(PWD)/venv
PYTHON   ?= python3
PIP      := $(VENV_DIR)/bin/pip
ACTIVATE := source $(VENV_DIR)/bin/activate

# ------------------------------------------------
# Generic venv creation target
# ------------------------------------------------
$(VENV_DIR):
	@echo ">>> Creating venv at $(VENV_DIR)"
	$(PYTHON) -m venv $(VENV_DIR) --prompt torch-qat
	$(PIP) install --upgrade pip

# ------------------------------------------------
# CPU setup
# ------------------------------------------------
setup-cpu: $(VENV_DIR)
	@echo ">>> Installing CPU-only stack"
	$(PIP) install -r requirements-ml-cpu.txt
	@echo
	@echo ">>> Done. Activate with:"
	@echo "    source $(VENV_DIR)/bin/activate"

# ------------------------------------------------
# CUDA 12.4 setup
# ------------------------------------------------
setup-cu124: $(VENV_DIR)
	@echo ">>> Installing CUDA 12.4  stack"
	$(PIP) install -r requirements-ml-cu124.txt
	@echo
	@echo ">>> Done. Activate with:"
	@echo "    source $(VENV_DIR)/bin/activate"

# ------------------------------------------------
# Utilities
# ------------------------------------------------
activate:
	source $(VENV_DIR)/bin/activate

check:
	@$(ACTIVATE) && \
	python -c 'import torch, cvxpy as cp; \
	print("torch:", torch.__version__, "cuda?", torch.cuda.is_available()); \
	print("cvxpy:", cp.__version__);' \

clean:
	@echo ">>> Removing venv at $(VENV_DIR)"
	rm -rf $(VENV_DIR)

