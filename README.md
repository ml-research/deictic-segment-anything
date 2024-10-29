### *Refactoring is undegoing.
<!-- <p align="center">
  <img src="./imgs/deisam_logo_eye.png">
</p>  -->

# DeiSAM: Segment Anything with Deictic Prompting (NeurIPS 2024)
Hikaru Shindo, Manuel Brack, Gopika Sudhakaran, Devendra Singh Dhami, Patrick Schramowski, Kristian Kersting

[AI/ML Lab @ TU Darmstadt](https://ml-research.github.io/index.html)

<p align="left">
  <img src="./imgs/deisam_task.png", height=180>
</p>
We propose DeiSAM, which integrates large pre-trained neural networks with differentiable logic reasoners. Given a complex, textual segmentation description, DeiSAM leverages Large Language Models (LLMs) to generate first-order logic rules and performs differentiable forward reasoning on generated scene graphs.


# Install
[Dockerfile](.devcontainer/Dockerfile) is avaialbe in the [.devcontainer](.devcontainer) folder.

To install further dependencies, clone [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) and then:
<!-- and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) repositories, and then-->
<!-- in the [Grounded-Segment-Anything](./Grounded-Segment-Anything) folder,  -->
```
cd neumann/
pip install -e .
cd ../Grounded-Segment-Anything/
cd segment_anything
pip install -e .
cd ../GroundingDINO
pip install -e .
```

If an error appears regarding OpenCV (circular import), try:
```
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip uninstall opencv-contrib-python-headless
pip3 install opencv-contrib-python==4.5.5.62
```

Download vit model
```
wget https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth
```

# Dataset
**DeiVG datasets can be downloaded here
[link](https://hessenbox.tu-darmstadt.de/getlink/fiJwsDNjdY9HDrUMf3btjoHG/).** Please locate downloaded files to `data/` as follows (make sure you are in the home folder of this project):
```
mkdir data/
cd data
wget https://hessenbox.tu-darmstadt.de/dl/fiJwsDNjdY9HDrUMf3btjoHG/.dir -O deivg.zip
unzip deivg.zip
cd visual_genome
unzip by-id.zip
```


Please download Visual Genome images [link](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html), and locate downloaded files to `data/visual_genome/` as follows:
```
cd data/visual_genome
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip images.zip
unzip iamges2.zip
mv VG_100K_2/* VG_100K/
```


# Experiments
To solve DeiVG using DeiSAM:
```
python src/solve_deivg.py --api-key YOUR_OPENAI_API_KEY -c 1
python src/solve_deivg.py --api-key YOUR_OPENAI_API_KEY -c 2
python src/solve_deivg.py --api-key YOUR_OPENAI_API_KEY -c 3
```


The demonstration of learning can be performed by:
```
python src/learning_demo.py --api-key YOUR_OPENAI_API_KEY -c 1 -sgg VETO -su
python src/learning_demo.py --api-key YOUR_OPENAI_API_KEY -c 2 -sgg VETO -su
```
*Note that DeiSAM is esseitially a training-free model.* Learning here is a demonstration of the learning capability by gradients. The best performance will be always achieved by using the model with ground-truth scene graphs, which corresponds to `solve_deivg.py`. 
In other words, DeiSAM doesn't need to be trained when the scene graphs are availale. A future plan is to mitigate the case where scene graphs are not available.

<!--
# Experiments

## Prerequisites
Docker container is available in folder [.devcontainer](./.devcontainer/Dockerfile),
which is compatible with [packages](./pip_requirements.txt) (produced by pip freeze).
The main dependent packages are:
```
pytorch
torch-geometric
networkx
```
We used Python 3.8 for the experiments.
See [Dockerfile](.devcontainer/Dockerfile) for more details.

## Build a Docker container
Simply use VSCode to open the container, or build the container manually:
To run on machines without GPUs
```
cp .devcontainer/Dockerfile_nogpu ./Dockerfile
docker build -t neumann .
docker run -it -v <local path to the repository>:/neumann --name neumann neumann
```
For example, the local path could be: `/Users/username/Workspace/github/neumann`. The path is where this repository has been cloned.

For the GPU-equipped machines, use:
```
cp .devcontainer/Dockerfile ./Dockerfile
docker build -t neumann .
docker run -it -v <local path to the repository>:/neumann --name neumann neumann
```
To open the container on machines without GPUs using VSCode, run
```
cp .devcontainer/Dockerfile_nogpu .devcontainer/Dockerfile
```
and use the VSCode remotehost extension (recommended). -->


<!--
## Perform learning
For example, in the container, learning Kandinsky patterns on red triangle using the demo dataset can be performed:
```
cd /neumann
python3 src/train_neumann.py --dataset-type kandinsky --dataset red-triangle --num-objects 6 --batch-size 12 --no-cuda --epochs 30 --infer-step 4 --trial 5 --n-sample 10 --program-size 1  --max-var 6 --min-body-len 6 --pos-ratio 1.0 --neg-ratio 1.0
```
An exenplary log can be found [redtrianlge_log.txt](./logs/redtriangle_log.txt).

More scripts are available:

[Learning kandinsky/clevr-hans patterns](./scripts/solve_kandinsky_clevr.sh)

[Solving Behind-the-Scenes](./scripts/solve_behind-the-scenes.sh) -->

# Bibtex
```
@inproceedings{shindo24deisam,
  author       = {Hikaru Shindo and
                  Manuel Brack and
                  Gopika Sudhakaran and
                  Devendra Singh Dhami and
                  Patrick Schramowski and
                  Kristian Kersting},
  title        = {DeiSAM: Segment Anything with Deictic Prompting},
  booktitle    = {Proceedings of the Conference on Advances in Neural Information Processing Systems (NeurIPS)},
  year         = {2024},
}

```



# LICENSE
See [LICENSE](./LICENSE).

