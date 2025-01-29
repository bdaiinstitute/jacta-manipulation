## Planner high-level overview
Pesudo-code for the single-goal single start planner.

```
Planner(start_state, goal_state, iterations, termination_threshold):
  states <- {start_state}
  rewards <- {reward_function(start_state)}
  actions <- {}

  for i in range(iterations):
    selected_states <- node_selection(states)
    new_actions, new_states <- node_extension(selected_states)
    new_rewards <- reward_function(new_states)
    states <- states U new_states
    actions <- actions U new_actions
    rewards <- rewards U new_rewards
      if mininimum over all state[ distance(state, goal_state) ] < termination_threshold:
        break
  return states, actions
```

## Node selection
Pesudo-code for the node selection.

## Node extension
Pesudo-code for the node extension.

## Reward function
Pesudo-code for the Reward function.
