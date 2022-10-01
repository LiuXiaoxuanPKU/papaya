for f in *.log; do
    mv -- "$f" "${f#utilization_}"
done