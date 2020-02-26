from robot_env import RobEnv
from agent import Agent
from sample import generate_examples
from action import ALL_ACTIONS, RobState
from data_generation import get_supervised_batchsize, GenData
from util import *
import action

import time

def get_rollout(env, actions, max_iter):
    """
    get the rollout from a series of actions
    """
    trace = []
    s = env.reset()

    for i in range(max_iter):
        a = action.Commit() if i >= len(actions) else actions[i]
        state, reward, done = env.step(a)
        trace.append((s, a, reward, state, done))
        s = state
        if done: break

    return trace

def get_supervised_sample(render_kind={'render_past_actions' : False}):
    prog, inputs, outputs = generate_examples(N_IO)
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
    if USE_PARALLEL:
        dataqueue = GenData(lambda: get_supervised_sample(),
                            n_processes=N_PROCESSES,
                            batchsize=BATCH_SIZE,
                            max_size=100)

    for i, (states, actions) in enumerate(
            dataqueue.batchIterator() if USE_PARALLEL else \
            get_supervised_batchsize(lambda: get_supervised_sample(),
                                     batchsize = BATCH_SIZE)):

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
        tot_start = time.time()
        envs = []
        for _ in range(N_ENVS_PER_ROLLOUT):
            _, inputs, outputs = generate_examples(N_IO)
            env = RobEnv(inputs, outputs)
            envs.append(env)

        traces = agent.get_rollouts(envs, N_ROLLOUTS)
        states, rewards, reward_actions, reward_states = sample_from_traces(traces)
        loss = agent.value_fun_optim_step(states, rewards)

        # train the policy network
        if len(reward_states) > 0:
            ploss = agent.learn_supervised(reward_states, reward_actions)
            ploss = ploss.item()
        else:
            ploss = 0.0

        if i % PRINT_FREQ == 0 and i != 0:
            print(f"Iteration: {i}, Value loss: {loss.item()}, \
                  Policy loss: {ploss}, Total time: {tot_end - tot_start}")
            agent.save(SAVE_PATH)

        tot_end = time.time()

def initialize_value_as_policy(agent):
    policy_params = agent.nn.named_parameters()
    value_params = agent.Vnn.named_parameters()

    for name, param in policy_params:
        if name in dict_value_params and not "action_decoder" in name:
            print(f"copying {name}")
            dict_value_params[name].data.copy_(param.data)
        else:
            print(f"skipped {name}")

if __name__ == '__main__':
    #load model or create model
    agent = Agent(ALL_ACTIONS)
    try:
        agent.load(LOAD_PATH, policy_only =  True)
        print("loaded model")
    except FileNotFoundError:
        print ("no saved model found ... training from scratch")
    #train
    train_supervised(agent)

    #optionally, can do this:
    initialize_value_as_policy(agent)

    #rl train, whatever that entails
    train_rl(agent)
