# Haruspex

Haruspex is a Convolutional Neural Network capable of recognizing and predicting secondary structure elements and nucleotides in Cryo-EM reconstruction density.

There are three ways of installing and using Haruspex:
 * [Docker (recommended)](#haruspex-docker)
 * [Anaconda (GPU-Only)](#haruspex-conda)
 * Manual installation (discouraged)

For more information on what to do with the output, see the [results section](#haruspex-results)


## Haruspex-Docker

### Installation

1. Install docker as described in the [official documentation](https://docs.docker.com/install/) by either using your systems package manager or compiling it from source.
   
2. Start the docker service.
   * `sudo systemctl start docker`
   
3. (OPTIONAL) Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) for Nvidia GPU support (requires CUDA).

4. Download repository.
   * `git clone https://gitlab.com/phimos/haruspex`
   * `cd haruspex`

5. Build the docker container.   
   Remember that you **must** be root or use sudo if you are not a member of the docker group.   
   (You can become a member of the docker group using `sudo usermod -a -G docker exampleUser`.)
   * CPU: `docker build -t haruspex --network=host . -f docker/Dockerfile`
   * GPU: `docker build -t haruspex --network=host . -f docker/Dockerfile_gpu`
   * **Remember the container ID**
  
6. Test the docker container.
   * CPU: `docker run haruspex:latest --help`
   * GPU: `nvidia-docker run haruspex:latest --help`

### Predicting Maps

Please note that depending on the source MRC file the prediction can be highly memory intensive.

1. Create a volume folder for docker; it will serve as the exchange point of input and output data. Copy the source MRC file.
   * `mkdir exchange`
   * `cp relion_filtered.mrc.gz exchange/relion_filtered.mrc.gz`

2. Run the prediction on the given exchange folder and source MRC file.
   * CPU: `docker run -v /home/user/haruspex/exchange/:/volume haruspex:latest -d map-predict /volume/relion_filtered.mrc.gz -o /volume`
   * GPU: `nvidia-docker run -v /home/user/haruspex/exchange/:/volume haruspex:latest -d map-predict /volume/relion_filtered.mrc.gz -o /volume`

3. Your results are stored in `exchange` as MRC and NPZ files. [See the results section for more info.](#haruspex-results)


## Haruspex-Conda

### Installation

1. Install [anaconda](https://www.anaconda.com/distribution/) as described on the website.  
   **NOTE**: All further commands such as `conda` or `activate` refer to the anaconda binaries from your installation.
You may need to specify their full path for them to work (e.g. `anaconda/bin/activate`).

2. Download repository.
   * `git clone https://gitlab.com/phimos/haruspex`
   * `cd haruspex`

3. Create an anaconda environment using `conda`.  
   **NOTE**: There are **TWO** ways of doing this, either is fine.
   * `conda create --name hpx --file conda/conda_requirements.txt`  
   **OR**
   * `conda env create -f conda/conda_haruspex_190116.yml`


### Predicting Maps

1. Enter the ~~Matrix~~ conda hpx environment.
   * `source activate hpx`
   * **NOTE**: you can leave the environment using `deactivate`

2. Run Haruspex
   * `cd haruspex`
   * `source/hpx_unet_190116.py -n network/hpx_190116 -d map-predict /path/to/your/relion_filtered.mrc.gz -o /your/output/directory/`

3. Your results are stored in `/your/output/directory/` as MRC and NPZ files. [See the results section for more info.](#haruspex-results)



# Haruspex Results

Haruspex will output it's predictions as MRC files with each class having it's own MRC file (E.g. `relion_filtered_helix.mrc`).  
For further automatic processing using numpy, a NPZ file containing all classes is also created. 
You can open the MRC files just like any other MRC files using Coot or UCSF Chimera.
Output MRC files are uncompressed.



