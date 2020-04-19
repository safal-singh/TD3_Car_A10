# TD3_Car_A10

State dimensions (observation space) - 
  1. Image cropped around car, from the contour image, with height and width of 100px. Later resized to 28x28 in actor and critic networks.
  2. Orientation wrt the target (1420, 622)
  3. Distance from the target
  
Action dimension - 
  1. Degrees to rotate the car
  2. Speed of the car
  
Image is encoded into 20-length array and then concatenated with orientation and distance in the actor network. Same encoding architecture is used for Critic network before concatenation with other state and action dimensions.

Episode criterion - End the episode whenever car reaches the target (within distance of 25px). The car is then assigned origin from the coordinates - (1164, 614), (575, 535), (241, 559), (134, 280), (710, 227), (1155, 256) - after the episode has ended.

Reward Criterion - 
  1. +5 when car reaches target
  2. +0.1 if car moves along the target
  3. -0.2 if car moves away from the target
  4. -0.5 if car reaches within 5px distance from any of the boundaries 
