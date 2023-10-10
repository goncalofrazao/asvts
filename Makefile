
PYTHON=python3
OUTPUT_DIR=files

FILES=$(shell ls *.in)
PUBLIC1_FILES=$(shell ls public1/*.dat)
PUBLIC2_FILES=$(shell ls public2/*.dat)
PATHS_FILES=$(shell ls public1/*.plan)


costs:
	for file in ${PATHS_FILES}; do $(PYTHON) main.py $$file -c; done

test:
	for file in ${PUBLIC2_FILES}; do $(PYTHON) main.py $$file; done


time:
	for file in ${PUBLIC2_FILES}; do time $(PYTHON) main.py $$file; done

gen:
	for i in {00..99}; do $(PYTHON) map_generator.py file$$(printf "%02d" $$i).in 4 5 2; done

run:
	for i in {00..99}; do $(PYTHON) main.py file$$(printf "%02d" $$i).in >> file$$(printf "%02d" $$i).out; done

public:
	for file in ${PUBLIC1_FILES}; do $(PYTHON) main.py $$file >> $$file.out; done

CLEAN:
	rm -rf *.in *.out
