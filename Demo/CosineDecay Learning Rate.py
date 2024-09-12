import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Set parameters for cosine decay
initial_learning_rate = 0.01
decay_steps = 1000  # Number of steps for the learning rate to decay over
alpha = 0.001       # Minimum learning rate at the end of decay

# Create a CosineDecay schedule
cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=alpha
)

# Simulate the learning rate schedule
steps = np.arange(0, decay_steps)
learning_rates = [cosine_decay(step).numpy() for step in steps]

# Plot the learning rate
plt.plot(steps, learning_rates)
plt.title('Cosine Annealing Learning Rate')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.grid(True)  # Optional: Add grid for better readability
plt.show()


# 1. What Happens After 1,000 Steps?
# When you use tf.keras.optimizers.schedules.CosineDecay, the learning rate follows a cosine 
# curve and decays toward the minimum learning rate (defined by alpha) over the specified 
# decay_steps. Once it reaches the decay_steps, the learning rate will typically stay at or 
# very close to the minimum value.

# So, if the total number of steps is more than 1,000 (for example, 1,001 or 2,000 steps), 
# the learning rate will remain at the minimum value defined by alpha for any step greater 
# than decay_steps. It will not reset the cosine function on its own, and the learning rate 
# won’t rise again.




import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# CosineDecayRestarts with 1000 steps per cycle
cosine_decay_restarts = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=0.01,  # Initial LR
    first_decay_steps=1000,      # 1000 steps per decay cycle
    t_mul=1.0,                   # Each restart cycle is the same length
    m_mul=1.0,                   # Initial LR stays the same after each restart
    alpha=0.001                  # Minimum LR after decay
)

# Simulate the learning rate schedule for 2000 steps
steps = np.arange(0, 2000)
learning_rates = [cosine_decay_restarts(step).numpy() for step in steps]

# Plot the learning rate schedule
plt.plot(steps, learning_rates)
plt.title('Cosine Annealing Learning Rate with Restarts')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.grid(True)
plt.show()

# CosineDecayRestarts Purpose:
# The key idea of CosineDecayRestarts is to allow the model to "escape" local minima by periodically increasing the learning rate (to 0.01 in this case) and then gradually decaying it again. By doing so:

# At the beginning of a restart, the learning rate starts high (e.g., 0.01) to allow the 
# model to explore larger changes in the loss landscape.

# As it decays, the learning rate becomes small again (e.g., 0.001) to fine-tune the weights.

# This controlled behavior helps avoid the model getting stuck in local minima and allows 
# it to continue learning and improving.