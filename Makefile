
PYTHON=python3
OUTPUT_DIR=files

FILES=$(shell ls *.in)

GEN:
	for i in {00..99}; do $(PYTHON) map_generator.py file$$(printf "%02d" $$i).in 4 5 2; done

RUN:
	for i in {00..99}; do $(PYTHON) main.py file$$(printf "%02d" $$i).in >> file$$(printf "%02d" $$i).out; done

CLEAN:
	rm -rf *.in *.out
