# Introduction
This example demostrates how to use ActionConditionalVideoPredcitionModel to predict next frame conditions on current state and action.
```example.npy``` is a [84, 84, 12] numpy array (DQN settings). 

# Usage
```
python example.py --load {tensorflow model checkpoint} --data {state npy file(e.g. s_t_.npy)} --mean {image mean} --num_act {number of action in the action space} --act {which action you want to take, 0 <= act < num_act}
```

# Integrate with your code
```
from tfacvp.model import ActionConditionalVideoPredictionModel
from tfacvp.util import post_process_gray, pre_process_gray

mean = np.load(meanfile_path)

sess = tf.Session()

model = ActionConditionalVideoPredictionModel(num_act=num_act, is_train=False)
model.restore(sess, chekckpoint_path)

s = pre_process_gray(s, mean, scale)
model.predict(sess, s, a)
```
