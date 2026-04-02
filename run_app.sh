#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
streamlit run app.py
