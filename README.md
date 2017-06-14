# rl-adversarial-attack-detection

### Dependencies
- baselines

**Change directory**
```
cd rl-adversarial-attack-detection/baselines
```

**Download pretrained models**
```
python -m baselines.deepq.experiments.atari.download_model --blob model-atari-duel-pong-1 --model-dir /tmp/models
```

** Collect data**
```
python -m baselines.deepq.experiments.atari.collect --model-dir /tmp/models/model-atari-duel-pong-1 --env Pong --dueling
```
