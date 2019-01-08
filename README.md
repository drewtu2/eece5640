log in:
`ssh -X tu.a@login.discovery.neu.edu`.  Please replace yourusername with your myneuid.  If you're on a mac use ssh -Y (not ssh -X)

submit test jobs:
`srun -p gpu --reservation=eece5640 --x11 --gres=gpu:1 nvidia-smi`
this will return information about the gpu on the node which was allocated for your job

`srun -p gpu --reservation=eece5640 --x11 --gres=gpu:1 --pty /bin/bash`
this will log you into an interactive shell on a node in the reservation.  From there you can run any linux commands

`--x11` is necessary only if you run any programs with a graphical interface that will display to your desktop.

if you need more info please check the RC website: http://neu.edu/rc
if you still need more info please email us at researchcomputing@northeastern.edu
