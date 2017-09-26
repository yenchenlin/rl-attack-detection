# rl-attack-detection

**DISCLAIMER**: This repository is a modified version of [openai/baselines](https://github.com/openai/gym).


### Dependencies
- Python 3
- cleverhans

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
Here I'll use game Freeway as an example to demonstrate how to run the code.

**1. Download pre-trained agent**

Download the repository contains pre-trained DQN agents for Freeway [here](https://drive.google.com/open?id=0B50cbskLVq-eRzBtNktCVE1SSms) to `rl-attack-detection/pre-trained-agents/`.

**2. Run pre-trained agent**

Test the performance of the pre-trained agent:

```
python -m baselines.deepq.experiments.atari.enjoy --model-dir ./pre-trained-agents/Freeway --env Freeway
```

For game Freeway, you should see output similar to follows:

```
29.0
27.0
28.0
...
```

**3. Perform adversarial attack**

Use adversarial example crafted by FGSM to attack deep RL agent:

```
python -m baselines.deepq.experiments.atari.enjoy --model-dir ./pre-trained-agents/Freeway --env Freeway --attack fgsm
```

**3. Perform adversarial attack**

