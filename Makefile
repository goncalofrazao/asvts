
PYTHON=python3

PUBLIC1_FILES=$(shell ls public1/*.dat)
PUBLIC2_FILES=$(shell ls public2/*.dat)
PUBLIC3_FILES=$(shell ls public3/*.dat)

pub3:
	for file in ${PUBLIC3_FILES}; do $(PYTHON) main.py $$file; done

time3:
	for file in ${PUBLIC3_FILES}; do /usr/bin/time $(PYTHON) main.py $$file; done

pub2:
	for file in ${PUBLIC2_FILES}; do $(PYTHON) main.py $$file; done

time2:
	for file in ${PUBLIC2_FILES}; do /usr/bin/time $(PYTHON) main.py $$file; done

pub1:
	for file in ${PUBLIC1_FILES}; do $(PYTHON) main.py $$file >> $$file.out; done

time1:
	for file in ${PUBLIC1_FILES}; do /usr/bin/time $(PYTHON) main.py $$file >> $$file.out; done