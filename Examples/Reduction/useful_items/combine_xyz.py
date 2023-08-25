import glob

FF_files = glob.glob('./FF_step*.xyz')
n = len(FF_files)
FF_files = ['FF_step'+str(i)+'.xyz' for i in range(1,n+1)]
VASP_files = glob.glob('./VASP_step*.xyz')
n = len(VASP_files)
VASP_files = ['VASP_step'+str(i)+'.xyz' for i in range(1,n+1)]

contents = []
contents += [open(FF_file,'r').read() for FF_file in FF_files]
contents += [open(VASP_file,'r').read() for VASP_file in VASP_files]

contents = ''.join([content for content in contents])
open('all_step.xyz','w').write(contents)
