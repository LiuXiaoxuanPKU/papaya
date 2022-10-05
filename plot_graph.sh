version=$(python3 -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$version" ]]
then
    version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
    if [[ -z "$version" ]]
    then
        python -W ignore ./experiment.py --machine-tag v100 --plot-graph # --algos gpt bert
        #python -W ignore ./experiment.py --machine-tag t4 --plot-graph --algos bert
    else
     echo "No python found."
    fi

else
    python3 -W ignore ./experiment.py --machine-tag v100 --plot-graph # --algos gpt bert
    #python3 -W ignore ./experiment.py --machine-tag t4 --plot-graph --algos bert
    #t4 bert missing data 
fi