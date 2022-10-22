#!/bin/bash
echo "Installing papaya customized fairseq..."
echo "Current pip: $(which pip)"
if pip install --editable ./fairseq/; then
    echo "Customized fairseq sucessfully installed."
else
    echo "Installation Failed."
fi
