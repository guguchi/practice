import subprocess
import math
import sys
import argparse 
import json
import time
import datetime
import os
import re

class slurm_tools:
	@staticmethod
	def bash_run(command):
		proc = subprocess.Popen(['/bin/bash'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
		return proc.communicate(command) , (proc.returncode == 0)
	      
	@staticmethod
	def slurm_submit(commands,**kwargs):
		params={};
		params['debug']=False;
		params['output']=None;
		params['mem']=5000;
		params['cores']=2;
		params['append']=False;
		params['time']='01-00:00:00';
		params['queue']='gpucpu';
		params['feature']='';
		params['export']='ALL';
		params['gres']='';
		if kwargs is not None:
			for key, value in kwargs.iteritems():
				params[key]=value;

		if not "error" in params:
		    params['error']=params['output'];

		

		batch='printf "';
		batch=batch+'#!/bin/bash \\n';

		if 'name' in params:
		  params['name']=params['name'].replace(' ','-');
		  batch=batch+'#SBATCH --job-name=\"'+format(params['name'])+'\"\\n';
		
		batch=batch+'#SBATCH -c '+format(params['cores'])+'\\n';
		batch=batch+'#SBATCH --mem '+format(params['mem'])+'\\n';
		batch=batch+'#SBATCH -t '+format(params['time'])+'\\n';
		batch=batch+'#SBATCH --export '+format(params['export'])+'\\n';
		if (params['error']!=None):
		  batch=batch+'#SBATCH --error '+params['error']+'\\n';
		if (params['output']!=None):
		  batch=batch+'#SBATCH --output '+params['output']+'\\n';
		batch=batch+'#SBATCH -p '+format(params['queue'])+'\\n';
		if len(params['feature'])>0:
			batch=batch+'#SBATCH --constraint=\"'+format(params['feature'])+'\"\\n';

		if len(params['gres'])>0:
			batch=batch+'#SBATCH --gres=\"'+format(params['gres'])+'\"\\n';

		if params['append']:
			batch=batch+'#SBATCH --open-mode append\\n';
		else:
			batch=batch+'#SBATCH --open-mode truncate\\n';

		for command in commands:
			command.replace('"','\"');
			batch=batch+command+'\\n';

		batch=batch+'" | sbatch';

		if 	params['debug']:
			batch="";
			for command in commands:
				command.replace('"','\"');
				batch=batch+command+';';
				res, success=slurm_tools.bash_run(batch);
				res="0";				
	
		else:
			print "#I: BATCH SCRIPT: ---------------------------------"
			print batch
			print "#I: -----------------------------------------------"
			res, success=slurm_tools.bash_run(batch);
			if success:
				res=res[0].split()[-1];
	

		return res, success;

