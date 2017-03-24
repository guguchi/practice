#!/usr/bin/python
from slurm import slurm_tools

for :
	SLURM_commands=['python /home/main.py {} {}'.format()];
	res, success=slurm_tools.slurm_submit(SLURM_commands,name="hello",mem=5000)

	print "job number: "+res
