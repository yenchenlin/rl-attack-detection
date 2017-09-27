# rl-attack-detection

**DISCLAIMER**: This repository is a modified version of [openai/baselines](https://github.com/openai/gym).


## Dependencies
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


## Example
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
python -m baselines.deepq.experiments.atari.enjoy --model-dir ./pre-trained-agents/Freeway --env Freeway --attack fgsm
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

Use visual foresight to protect deep RL agent:

```
python -m baselines.deepq.experiments.atari.enjoy --model-dir ./pre-trained-agents/Freeway --env Freeway --attack fgsm --defense foresight
```

Now, you should see similar outputs to **step. 2**, which means that now it works again.
