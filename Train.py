from RobState import RobState
from RobEnv import RobEnv
from Agent import Agent
from Sample import generate_examples
from GenData import get_supervised_batchsize, GenData, makeTestdata
from Util import *
import Action

def get_rollout(env, actions, max_iter):
    """
    get the rollout from a series of actions
    """
    trace = []
    s = env.reset()

    for i in range(max_iter):
        a = Action.Commit() if i >= len(actions) else actions[i]
        state, reward, done = env.step(a)
        trace.append((s, a, reward, state, done))
        s = state
        if done: break

    return trace

def get_supervised_sample(render_kind={'render_past_actions' : False}):
    prog, inputs, outputs = generate_examples(N_IOS)
    env = RobEnv(inputs, outputs, render_kind)
    trace = get_rollout(env, prog.to_action(), 30)

    states = [x[0] for x in trace]
    actions = [x[1] for x in trace]
    return states, actions

def get_supervised_batch(batch_size = 200):
    remains = [], []

    while True:
        pre_states, pre_actions = remains
        states, actions = get_supervised_sample()
        states, actions = pre_states + states, pre_actions + actions

        curr_batch_size = len(states)
        if curr_batch_size >= batch_size:
            yield states[:batch_size], actions[:batch_size]
            remains = states[batch_size:], actions[batch_size:]
        else:
            remains = states, actions

def sample_from_traces(traces):
    states = []
    rewards = []
    actions = []
    prev_states = []

    # each trace entry has the format
    # (prev_state, action, reward, curr_state, done)
    for trace in traces:
        r = trace[-1][2]
        pstates, acts, rs, cstates, _ = list(zip(*trace))
        states += cstates
        rewards += rs
        if r == 1.0:
            actions += acts
            prev_states += pstates

    return states, rewards, actions, prev_states

def train_supervised(agent):
    for i, (states, actions) in enumerate(get_supervised_batch()):
        if agent.train_iterations >= TRAIN_ITERATIONS:
           break

        # supervised learning for the policy network
        loss = agent.learn_supervised(states, actions)

        if i % PRINT_FREQ == 0:
            print("========info========")
            print(f'Iteration {i}, Loss {loss}')

        if i % SAVE_FREQ == 0 and i != 0:
            print("========saving========")
            agent.save(SAVE_PATH)

        if i % TEST_FREQ == 0 and i != 0:
            print("========testing========")
            states, actions = get_supervised_sample()
            sample_actions = agent.sample_actions(states)
            print("========real actions========")
            print(actions)
            print("========model actions========")
            print(sample_actions)

        if hasattr(agent, 'train_iterations'):
            agent.train_iterations += 1
        else:
            agent.train_iterations = 1

    agent.save(SAVE_PATH)

def train_rl(agent):
    for i in range(RL_ITERATIONS):
        envs = []
        for _ in range(N_ENVS_PER_ROLLOUT):
            _, inputs, outputs = generate_examples(N_IO)
            env = RobEnv(inputs, outputs)
            envs.append(env)

        traces = agent.get_rollouts(envs, N_ROLLOUTS)
        states, rewards, past_actions, prev_states = sample_from_traces(traces)
        

