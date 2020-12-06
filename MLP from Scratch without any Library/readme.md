## BackGround

There is a Lander Game, In which by using the arrow keys we need to land the aircraft to a perticular target area. For each game the position of target changes.
Based on the data collected, we want to train the neural network to predict the new velocities and land the aircraft in the target by its own.

## Data Collection:
Play the game in Data collection mode and collect the data needed.
Below is the data collected:

1. X Distance to Target
2. Y Distance to Target
3. New Velocity Y
4. New Velocity X

we need to train the neural network to predict New Velocity Y,New Velocity X based on X Distance to Target & Y Distance to Target

We will use Feed Forward Back Propagation with Momentum
