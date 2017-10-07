# Deep Q Learning
bare bones example of deep q learning with openai's frozenlake (variant of gridworld).

## what is deep q learning?
dqn uses a deep neural network to approximate a Q function, which, for a given state-action pair, returns a set of Q values for each possible action. you can think of a Q value as the maximum possible sum of discounted rewards, assuming the agent takes the optimal path to the end.

## actor/critic
rather than have a single agent predict value and policy, state of the art separates learning into actor/critic. 

the critic (value network) predicts a single value for each state. the critic value is completely unaware of actions when it learns.

the actor (policy network) predicts a set of deltas (difference in values between the current state and next states). as result, it learns to select actions that result in the biggest increase in value, rather than the action that leads to the state with the biggest value.
