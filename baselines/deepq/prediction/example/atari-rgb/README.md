# Introduction
This example use MsPacman-v0 for demostrating infer next frame with current four frames and action.
```example.npy``` is a [84, 84, 12] numpy array (DQN settings). 

# Usage
```
python example.py --load {tensorflow model checkpoint}
```

# Integrate with your code
```
from tfacvp.model import ActionConditionalVideoPredictionModel
from tfacvp.util import post_process 

model = ActionConditionalVideoPredictionModel(num_act=num_act, is_train=False)
sess = tf.Session()
model.restore(sess, args.load)
model.predict(sess, s, a)
```
