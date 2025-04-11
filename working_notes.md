1. pip process bug

```
(gr00t) jojo@jojo-System-Product-Name:~/Documents/Isaac-GR00T$ pip install --no-build-isolation flash-attn==2.7.1.post4 
Collecting flash-attn==2.7.1.post4
  Downloading flash_attn-2.7.1.post4.tar.gz (2.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.7/2.7 MB 17.2 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [22 lines of output]
      fatal: not a git repository (or any of the parent directories): .git
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-a7j6c1sr/flash-attn_a8b229a6259a4d9f9626e437799d8fe1/setup.py", line 163, in <module>
          _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        File "/tmp/pip-install-a7j6c1sr/flash-attn_a8b229a6259a4d9f9626e437799d8fe1/setup.py", line 82, in get_cuda_bare_metal_version
          raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
        File "/home/jojo/anaconda3/envs/gr00t/lib/python3.10/subprocess.py", line 421, in check_output
          return run(*popenargs, stdout=PIPE, timeout=timeout, check=True,
        File "/home/jojo/anaconda3/envs/gr00t/lib/python3.10/subprocess.py", line 503, in run
          with Popen(*popenargs, **kwargs) as process:
        File "/home/jojo/anaconda3/envs/gr00t/lib/python3.10/subprocess.py", line 971, in __init__
          self._execute_child(args, executable, preexec_fn, close_fds,
        File "/home/jojo/anaconda3/envs/gr00t/lib/python3.10/subprocess.py", line 1863, in _execute_child
          raise child_exception_type(errno_num, err_msg, err_filename)
      FileNotFoundError: [Errno 2] No such file or directory: ':/usr/local/cuda-12.4/bin/nvcc'
      
      
      torch.__version__  = 2.5.1+cu124
      
      
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```