from godot_env      import GodotEnv
from sarsa_agent    import SARSAAgent
import csv
import os

NUM_EPISODES    = 1000
MAX_STEPS       = 1000

env = GodotEnv()
agent = SARSAAgent()

log_rows = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    action = agent.select_action(state)
    total_reward = 0
    steps = 0

    for step in range(MAX_STEPS):
        next_state, reward, done, score = env.step(action)
        next_action = agent.select_action(next_state)

        agent.update(state,action,reward,next_state,next_action,done)

        state = next_state
        action = next_action
        total_reward += reward
        steps += 1

        if done: break

    agent.decay_epsilon()
    log_rows.append([episode,total_reward,score,steps,agent.epsilon])

    if episode % 10 == 0:
        print(f"Episode {episode}: reward={total_reward:.2f} score={score} steps={steps} epsilon={agent.epsilon:.3f}")

env.close()

log_filename = "training_log.csv"
counter = 1
while os.path.exists(log_filename):
    log_filename = f"training_log({counter}).csv"
    counter += 1

with open(log_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["episode", "total_reward", "score", "steps", "epsilon"])
    writer.writerows(log_rows)

print(f"Training complete. Log saved to {log_filename}")