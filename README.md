# rl-adversarial-attack-detection

### Dependencies
- cleverhans
- baselines

### DQN
first change directory:
```
cd baselines
```

**Experiment attack detection**


**Experiment action suggestion**
Normal agent:
```
python -m baselines.deepq.experiments.atari.enjoy --model-dir /tmp/models/model-you-want --env game-you-chose --dueling
```
FGSM attacked agent:
```
python -m baselines.deepq.experiments.atari.enjoy --model-dir /tmp/models/model-you-want --env game-you-chose --dueling --attack FGSM
```
Carlini attacked agent:
```
python -m baselines.deepq.experiments.atari.enjoy --model-dir /tmp/models/model-you-want --env game-you-chose --dueling --attack FGSM
```

**Download pretrained models**
There's a variety of models to choose from. You can list them all by running:
```
python -m baselines.deepq.experiments.atari.download_model
```

Once you pick a model, you can download it and visualize the learned policy. Be sure to pass --dueling flag to visualization script when using dueling models.
```
python -m baselines.deepq.experiments.atari.download_model --blob model-you-want --model-dir /tmp/models
python -m baselines.deepq.experiments.atari.enjoy --model-dir /tmp/models/model-you-want --env game-you-chose --dueling
```
