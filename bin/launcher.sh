#!/bin/bash

# --- Step 1: Initialize Conda ---
# This makes the 'conda' command available to the script.
# It's the same line from your .bashrc.
if [ -f /usr/prog/miniconda/bin/activate ]; then
    source /usr/prog/miniconda/bin/activate
fi

# --- Step 2: Activate Your Specific Environment ---
# This loads the correct Python interpreter and libraries.
conda activate talib_yf_pygame

# --- Step 3: Set the PYTHONPATH ---
# This tells Python where to find your custom modules like data_retrieval.py.
#export PYTHONPATH="$HOME/SCRIPTS/FINANCE_SCRIPTS:$PYTHONPATH"
#export PYTHONPATH="/usr/prog/SCRIPTS/FINANCE_SCRIPTS:$PYTHONPATH"
export PYTHONPATH="/usr/prog/STEGO/lib::$PYTHONPATH"

# --- Step 4: Change to the Script's Directory ---
# This ensures any relative file paths in your scripts work correctly.
#cd "/home/groggs/SCRIPTS/FINANCE_SCRIPTS/STEGO_OCT_28_2025"
#cd "/usr/prog/SCRIPTS/FINANCE_SCRIPTS/STEGO"
cd "/usr/prog/STEGO/bin"

# --- Step 5: Run the Python GUI ---
# This now runs with the correct Conda environment and PYTHONPATH.
python3 gui.launcher.V4.py


