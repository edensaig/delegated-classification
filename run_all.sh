#! /bin/bash
set -e
start_time=$(date +%s)
for f in ./[a-z]*.ipynb; do
	echo ===================
	echo $f
	ipython --TerminalIPythonApp.file_to_run=$f
	echo
done
end_time=$(date +%s)
echo Done! Wall time: $((end_time-start_time))s