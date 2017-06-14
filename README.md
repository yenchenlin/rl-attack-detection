# rl-adversarial-attack-detection

### Dependencies
- baselines

**Clone repo**
```
git clone git@github.com:yenchenlin/rl-adversarial-attack-detection.git
```

**Change directory**
```
cd rl-adversarial-attack-detection/baselines
```

**Checkout branch**
```
git fetch
git checkout collect
```

**Download pretrained models**
```
python -m baselines.deepq.experiments.atari.download_model --blob model-atari-duel-pong-1 --model-dir /tmp/models
```

**Collect data**
```
python -m baselines.deepq.experiments.atari.collect --model-dir /tmp/models/model-atari-duel-pong-1 --env Pong --dueling
```
