### Instructions for Interactive Job
- On the Perlmutter login node, to get one node: 
```
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m3957_g
```

To setup your environment once you're on the node, from the repository root, run: 
```
module load python/3.9-anaconda-2021.11
module load pytorch/1.10
conda create -p ./conda-env --clone /global/common/software/nersc/shasta2105/pytorch/1.10.0
conda activate conda-env/
pip install --upgrade pip setuptools wheel
pip install -r requirements/requirements.txt
pip install -e .
```

We are currently using a forked version of huggingface transformers library which fixes a problem with the Reformer model. Please clone our fork from: https://github.com/maxzvyagin/transformers

The other packages that the code depends on are easily installed using pip. 

### Instructions for Batch Job
This is how to run a distributed job at scale.

First load the environment:
```
module load python/3.9-anaconda-2021.11
module load pytorch/1.10
conda activate conda-env/
```
Navigate to your run directory which contains the configuration file
and then create an SBATCH script:

#### Example SBATCH script - submit.sh:

Then, from a login node, simply run 
```
sbatch submit.sh
```
and it'll be added to the queue. 
You can monitor job status using `sqs`.

Note: All config paths should be absolute paths or relative paths to the running directory.


### Note for Installing Deepspeed on Perlmutter - 6/9/2022
To build the DeepSpeed ops on perlmutter, you need a very specific compiler setup with gcc and nvcc. It doesn't work when you try to build it using the compilers that come standard on Perlmutter. The best way I've found to address this is by creating a conda environment to install these compilers into, then re-running the deepspeed install with DS_BUILD_OPS=1 and DS_BUILD_AIO=0. 

Conda environment list which led to successful build:
```
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main
_openmp_mutex             5.1                       1_gnu
_sysroot_linux-64_curr_repodata_hack 3                   haa98f57_10
binutils_impl_linux-64    2.36.1               h193b22a_2    conda-forge
binutils_linux-64         2.36                hf3e587d_10    conda-forge
gcc                       9.4.0               h192d537_10    conda-forge
gcc_impl_linux-64         9.4.0               h03d3576_16    conda-forge
gcc_linux-64              9.4.0               h391b98a_10    conda-forge
gxx_impl_linux-64         9.4.0               h03d3576_16    conda-forge
gxx_linux-64              9.4.0               h0316aca_10    conda-forge
kernel-headers_linux-64   2.6.32              he073ed8_15    conda-forge
ld_impl_linux-64          2.36.1               hea4e1c9_2    conda-forge
libgcc-devel_linux-64     9.4.0               hd854feb_16    conda-forge
libgcc-ng                 11.2.0               h1234567_1
libgomp                   11.2.0               h1234567_1
libsanitizer              9.4.0               h79bfe98_16    conda-forge
libstdcxx-devel_linux-64  9.4.0               hd854feb_16    conda-forge
libstdcxx-ng              11.2.0               h1234567_1
sysroot_linux-64          2.12                he073ed8_15    conda-forge
```
