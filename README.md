# aics-track-curator
Python script for track curation in Paraview. Only available for OSX at the moment.

# One time actions:

1. Download paraview from https://www.paraview.org/download/.

2. Clone this repo:

`git clone https://github.com/aics-int/aics-track-curator.git`

3. Create an environment

```
conda create --name curator python=3.9
conda activate curator
bash install.sh
```

# Usage:

Open Paraview and you should see a button named `curator` in the menu. To check for the available actions, plese click `curator`, type `help` and hit OK.

To create a PNG image with the curated lineage information go to the terminal and type:

```
conda activate curator
python ~/.config/ParaView/Macros/curator.py path/to/dataset/directory
```
