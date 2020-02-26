import torch
import torch.nn as nn
from namedtensor import ntorch
import torch.nn.functional as F

from util import *

class DenseLayer(nn.Module):
    """
    one layer in the dense network
    """
    def __init__(self, in_size, out_size):
        super().__init__()
        # self activation
        self.linear = ntorch.nn.Linear(in_size, out_size).spec('h', 'h')

    def forward(self, x):
        return self.linear(x).relu()

class DenseBlock(nn.Module):
    """
    a densely connected layers with skip connections
    """
    def __init__(self, num_of_layers, growth_rate, input_size, output_size):
        super().__init__()

        modules = [DenseLayer(input_size, growth_rate)]
        for i in range(1, num_of_layers - 1):
            modules.append(DenseLayer(input_size + i * growth_rate,
                                      growth_rate))
        modules.append(DenseLayer(input_size + (num_of_layers - 1) *
                                  growth_rate, growth_rate))

        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            output = layer(ntorch.cat(inputs, 'h'))
            inputs.append(output)
        return inputs[-1]

class Encoder(nn.Module):
    """
    encode a string into a network
    """

    def __init__(self):
        super().__init__()

        self.char_embedding = ntorch.nn.Embedding(len(CHARACTERS) + 1,
                                                  CHAR_EMBED_DIM).spec('stateLoc',
                                                                       'charEmb')
        # convolution
        self.column_encoding = ntorch.nn.Conv1d(
            # inputs, outputs, scratch, committed, mask
            in_channels = 4 * CHAR_EMBED_DIM + 1,
            out_channels = COLUMN_ENCODING_DIM,
            kernel_size = KERNEL_SIZE,
            padding = int((KERNEL_SIZE - 1) / 2)).spec('inFeatures',
                                                       'strLen',
                                                       'expression')

        # next layers are a dense network

        self.MLP = DenseBlock(DENSE_LAYERS,
                              GROWTH_RATE,
                              COLUMN_ENCODING_DIM * STR_LEN,
                              H_OUT)

    def forward(self, chars, masks):
        chars_emb = self.char_embedding(chars)
        chars_emb = chars_emb.stack(('charEmb', 'stateLoc'), 'inFeatures')

        x = ntorch.cat([chars_emb, masks], 'inFeatures')
        e = self.column_encoding(x)
        e = e.stack(('strLen', 'expression'), 'h')
        h = self.MLP(e)

        return h

class Model(nn.Module):
    def __init__(self, num_of_actions, is_value_net = False):
        super().__init__()
        self.is_value_net = is_value_net
        self.encoder = Encoder()
        self.action_embedding = ntorch.nn.Embedding(num_of_actions + 1,
                                                    ACTION_EMBED_DIM).spec('batch','h')
        # if encode past actions, add them here

        self.fc = ntorch.nn.Linear(H_OUT + ACTION_EMBED_DIM, H_OUT)
        # max pooling
        self.pooling = lambda x: x.max('Examples')[0]

        if is_value_net:
            self.action_decoder = ntorch.nn.Linear(H_OUT, 2).spec('h', 'value')
            self.loss_fn = ntorch.nn.NLLLoss().spec('value')
        else:
            self.action_decoder = ntorch.nn.Linear(H_OUT,
                                                   num_of_actions).spec('h',
                                                                        'actions')
            self.loss_fn = ntorch.nn.CrossEntropyLoss().spec('actions')

        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, chars, masks, last_actions):
        x = self.encoder(chars, masks)
        x = self.pooling(x)
        la = self.action_embedding(last_actions)
        x = ntorch.cat([x, la], 'h')
        x = self.fc(x).relu()
        x = self.action_decoder(x)

        if self.is_value_net:
            x = x._new(F.log_softmax(x._tensor, dim = x._schema.get('value')))

        return x

    def learn_supervised(self, chars, masks, last_actions, targets):
        self.opt.zero_grad()

        output_dists = self(chars, masks, last_actions)
        loss = self.loss_fn(output_dists, targets)
        loss.backward()
        self.opt.step()
        return loss

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path))

class Agent:
    def __init__(self, actions, use_cuda=None):
        self.actions = actions
        self.idx = {x.name: i for i, x in enumerate(actions)}
        self.name_to_action = {x.name: x for x in actions}
        self.idx_to_action = {self.idx[x.name]: self.name_to_action[x.name] for x in actions}

        self.use_cuda = use_cuda
        if use_cuda == None:
            self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.nn = Model(len(actions)).cuda()
            self.Vnn = Model(len(actions), is_value_net=True).cuda()
        else:
            self.nn = Model(len(actions))
            self.Vnn = Model(len(actions), is_value_net=True)

    def state_to_sensor(self, states):
        inputs, scratchs, committeds, outputs, masks, last_actions = zip(*states)

        inputs = np.stack(inputs)
        input_tensor = ntorch.tensor(inputs, ('batch',
                                              'Examples',
                                              'strLen'))
        scratchs = np.stack(scratchs)
        scratch_tensor = ntorch.tensor(scratchs, ('batch',
                                                  'Examples',
                                                  'strLen'))
        committeds = np.stack(committeds)
        committed_tensor = ntorch.tensor(committeds, ('batch',
                                                      'Examples',
                                                      'strLen'))
        outputs = np.stack(outputs)
        output_tensor = ntorch.tensor(outputs, ('batch',
                                                'Examples',
                                                'strLen'))
        chars = np.stack([inputs, scratchs, committeds, outputs], 'stateLoc')
        chars = chars.transpose('batch', 'Examples',
                                'strLen', 'stateLoc').long()

        masks = np.array([masks])
        masks = ntorch.tensor(masks, ('batch', 'Examples', 'inFeatures', 'strLen'))
        masks = masks.transpose('batch', 'Examples', 'strLen', 'inFeatures').float()

        last_actions = np.stack(last_actions)
        last_actions = ntorch.tensor(last_actions, 'batch').long()

        if self.use_cuda:
            return chars.cuda(), masks.cuda(), last_butts.cuda()
        else:
            return chars, masks, last_actions

    def sample_actions(self, states):
        chars, masks, last_actions = self.state_to_tensor(states)
        logits = self.nn.forward(chars, masks, last_actions)
        dist = ntorch.distributions.Categorical(logits=logits,
                                                dim_logit='actions')
        sample = dist.sample()
        action_list = [self.idx_to_action[sample[{'batch':i}].item()]
                       for i in range(sample.shape['batch'])]
        return action_list

    # not a symbolic state here
    # actions are 2, 3 instead of 0,1 index here
    def learn_supervised(self, states, actions):
        chars, masks, last_actions = self.states_to_tensors(states)
        targets = self.actions_to_target(actions)
        loss = self.nn.learn_supervised(chars, masks, last_actions, targets)
        return loss

    def value_fun_optim_step(self, states, rewards):
        chars, masks, last_actions = self.states_to_tensors(states)
        targets = self.rewards_to_target(rewards)
        #print("TARGETS",targets.sum("batch").item())
        loss = self.Vnn.learn_supervised(chars, masks, last_actions, targets)
        return loss

    def compute_values(self, states):
        chars, masks, last_actions = self.states_to_tensors(states)
        output_dists = self.Vnn(chars, masks, last_actions)
        return output_dists

    def actions_to_target(self, actions):
        indices = [self.idx[a.name] for a in actions]
        target = ntorch.tensor( indices, ('batch',) ).long()
        return target.cuda() if self.use_cuda else target

    def rewards_to_target(self, rewards):
        target = ntorch.tensor( rewards, ('batch',) ).long().relu()
        return target.cuda() if self.use_cuda else target

    def get_rollouts(self, initial_envs, n_rollouts = 1000, max_iter = 30):
        envs = []
        traces = []
        active_states = []
        n_initial_envs = len(initial_envs)

        # initialization, execute the actions simultaneously in several envs
        for env in initial_envs:
            env.reset()
            for _ in range(n_rollouts):
                e = env.copy()
                envs.append(e)
                traces.append([])
                active_states.append(env.last_step[0])

        for i in range(max_iter):
            if not i == 0:
                active_states = [t[-1].s for t in traces if not t[-1].done]
            action_list = self.sample_actions(active_states) if active_states else []
            # prevents nn running on nothing
            action_list_iter = iter(action_list)
            active_states_iter = iter(active_states)
            if action_list == []: return traces

            for j in range(n_initial_envs*n_rollouts):
                if i > 0 and traces[j][-1].done: #if done:
                    continue
                a = next(action_list_iter)
                ss, r, done = envs[j].step(a)
                if i == 0:
                    prev_s = envs[j].last_step[0]
                else:
                    prev_s = traces[j][-1][0]
                traces[j].append((prev_s, a, r, ss, done))

        return traces

    def save(self, save_path):
        self.nn.save(save_path)
        self.Vnn.save(save_path + 'vnet')

    def load(self, load_path, policy_only = False):
        self.nn.load(load_path)
        if not policy_only:
            self.Vnn.load(load_path + 'vnet')


