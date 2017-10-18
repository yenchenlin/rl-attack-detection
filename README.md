## Detecting Adversarial Attacks on Neural Network Policies with Visual Foresight

![](https://user-images.githubusercontent.com/7057863/30933455-9e86ba96-a398-11e7-87fa-d6339ad60c51.gif)

**DISCLAIMER**: This repository is a modified version of [openai/baselines](https://github.com/openai/gym).

### Publication

Paper: https://drive.google.com/file/d/0B50cbskLVq-ed2F3eUw4SWQxbUU/view

```
@article{Lin2017RLAttackDetection,
  title={Detecting Adversarial Attacks on Neural Network Policies with Visual Foresight},
  author={Lin, Yen-Chen and Liu, Ming-Yu and Sun, Min and Huang, Jia-Bin},
  journal={arXiv preprint arXiv:1710.00814},
  year={2017}
}
```


### Dependencies
- Python 3
- cleverhans v2.0.0

```
pip install -e git+http://github.com/tensorflow/cleverhans.git#egg=cleverhans
```

- others (e.g., gym, baselines, ...)

```
git clone https://github.com/yenchenlin/rl-attack-detection.git
cd rl-attack-detection
pip install -e .
```


### Example
Here I'll use Atari game Freeway as an example to demonstrate how to run the code.

Let's start by switch to the home directory:

```
cd rl-attack-detection
```

**1. Download pre-trained agent**

Download [this repository](https://drive.google.com/open?id=0B50cbskLVq-eRzBtNktCVE1SSms) which contains pre-trained DQN agents for Freeway to `./atari-pre-trained-agents/`.

**2. Run pre-trained agent**

Test the performance of the pre-trained agent:

```
python -m baselines.deepq.experiments.atari.enjoy --model-dir ./atari-pre-trained-agents/Freeway --env Freeway
```

For game Freeway, you should see output similar to follows:

```
29.0
27.0
28.0
...
```
This means that our agent is now a master of the game!

**3. Perform adversarial attack**

Use adversarial example crafted by FGSM to attack deep RL agent:

```
python -m baselines.deepq.experiments.atari.enjoy --model-dir ./atari-pre-trained-agents/Freeway --env Freeway --attack fgsm
```

**Other attacks:** argument passed to `--attack` can be `fgsm`, `iterative`, `cwl2`.


You should see output similar to follows:

```
0.0
0.0
0.0
...
```

which means that the agent is fooled by adversary and went crazy!

**4. Use visual foresight as defense**

To protect the agent, first download [this repository](https://drive.google.com/drive/folders/0B50cbskLVq-eTGxqNWtkSGJsRzQ) which contains pre-trained visual foresight module for Freeway to `./atari-visual-foresight/`.

Then, we can use visual foresight to protect deep RL agent:

```
python -m baselines.deepq.experiments.atari.enjoy --model-dir ./atari-pre-trained-agents/Freeway --env Freeway --attack fgsm --defense foresight
```

Now, you should see similar outputs to **step. 2**, which means that our agents work well again.

### Add More Attacks
To use new attack methods, you can add the attack code [here](https://github.com/yenchenlin/rl-attack-detection/blob/master/baselines/deepq/build_graph.py#L156).
Generally, attack methods that follow the interface of [cleverhans](https://github.com/tensorflow/cleverhans) can be added within few lines.
