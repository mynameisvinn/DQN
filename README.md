# Deep Q Learning
bare bones example of deep q learning with openai's frozenlake (a slightly more difficult variant of gridworld).

## what is deep q learning?
we'll use a deep neural network to approximate a Q function, which, for a given state-action pair, returns a set of Q values for each possible action. you can think of a Q value as the maximum possible sum of discounted rewards, assuming the agent takes the optimal path to the end.
