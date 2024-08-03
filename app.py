from flask import Flask, render_template, request, jsonify
from q_learning import QLearningAgent
from environment import RoboticArmEnv
import numpy as np

app = Flask(__name__)

# Initialize environment and agent
env = RoboticArmEnv()
agent = QLearningAgent(state_size=env.state_size, action_size=env.action_size)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    episodes = int(request.form.get('episodes', 1000))
    max_steps = int(request.form.get('max_steps', 50))
    for episode in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    return jsonify({'message': 'Training completed'})

@app.route('/get_q_table', methods=['GET'])
def get_q_table():
    return jsonify(agent.q_table.tolist())

if __name__ == '__main__':
    app.run(debug=True)
