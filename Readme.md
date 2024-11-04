# GNN Binary classification for AD vs CN using structural connectivity.

## Environment
- Download anaconda for python https://www.anaconda.com/download/
- Use python 3.12 to create environment(conda create --name p1_advm python=3.12 )
- conda activate p2_advm
- Install the required libraries:
    - conda install pytorch torchvision torchaudio pytorch-cuda=12.7(change as per your version) -c pytorch -c nvidia(change as per your gpu)
    - conda install conda-forge::matplotlib
    - conda install seaborn
- You can also install the libraries from pip using the requirements.txt in the current folder(pip install -r requirements.txt).


## Steps to Run
- python SimpleGCN.py

## Output
- The output are visible in command line.
- The plots are saved in results folder.