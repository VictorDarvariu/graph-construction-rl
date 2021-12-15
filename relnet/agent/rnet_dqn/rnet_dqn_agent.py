import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from relnet.agent.rnet_dqn.q_net import NStepQNet, greedy_actions
from relnet.agent.rnet_dqn.nstep_replay_mem import NstepReplayMem
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.utils.config_utils import get_device_placement


class RNetDQNAgent(PyTorchAgent):
    algorithm_name = "rnet_dqn"
    is_deterministic = False
    is_trainable = True

    def __init__(self, environment):
        super().__init__(environment)

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)
        self.setup_nets()
        self.take_snapshot()

    def train(self, train_g_list, validation_g_list, max_steps, **kwargs):
        self.setup_graphs(train_g_list, validation_g_list)
        self.setup_sample_idxes(len(train_g_list))

        self.setup_mem_pool(max_steps, self.hyperparams['mem_pool_to_steps_ratio'])
        self.setup_histories_file()
        self.setup_training_parameters(max_steps)

        pbar = tqdm(range(self.burn_in), unit='batch', disable=None)
        for p in pbar:
            with torch.no_grad():
                self.run_simulation()
        pbar = tqdm(range(max_steps + 1), unit='steps', disable=None)
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        for self.step in pbar:
            with torch.no_grad():
                self.run_simulation()
            if self.step % self.net_copy_interval == 0:
                self.take_snapshot()
            self.check_validation_loss(self.step, max_steps)

            cur_time, list_st, list_at, list_rt, list_s_primes, list_term = self.mem_pool.sample(
                batch_size=self.batch_size)
            list_target = torch.Tensor(list_rt)
            if get_device_placement() == 'GPU':
                list_target = list_target.cuda()

            cleaned_sp = []
            nonterms = []
            for i in range(len(list_st)):
                if not list_term[i]:
                    cleaned_sp.append(list_s_primes[i])
                    nonterms.append(i)

            if len(cleaned_sp):
                _, _, banned = zip(*cleaned_sp)
                _, q_t_plus_1, prefix_sum_prime = self.old_net((cur_time + 1) % 2, cleaned_sp, None)
                _, q_rhs = greedy_actions(q_t_plus_1, prefix_sum_prime, banned)
                list_target[nonterms] = q_rhs

            list_target = Variable(list_target.view(-1, 1))
            _, q_sa, _ = self.net(cur_time % 2, list_st, list_at)

            loss = F.mse_loss(q_sa, list_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('exp: %.5f, loss: %0.5f' % (self.eps, loss))

            should_stop = self.check_stopping_condition(self.step, max_steps)
            if should_stop:
                break

    def setup_nets(self):
        self.net = NStepQNet(self.hyperparams, num_steps=2)
        self.old_net = NStepQNet(self.hyperparams, num_steps=2)
        if get_device_placement() == 'GPU':
            self.net = self.net.cuda()
            self.old_net = self.old_net.cuda()
        if self.restore_model:
            self.restore_model_from_checkpoint()

    def setup_mem_pool(self, num_steps, mem_pool_to_steps_ratio):
        exp_replay_size = int(num_steps * mem_pool_to_steps_ratio)
        self.mem_pool = NstepReplayMem(memory_size=exp_replay_size, n_steps=2)

    def setup_training_parameters(self, max_steps):
        self.learning_rate = self.hyperparams['learning_rate']
        self.eps_start = self.hyperparams['epsilon_start']

        eps_step_denominator = self.hyperparams['eps_step_denominator'] if 'eps_step_denominator' in self.hyperparams else 2
        self.eps_step = max_steps / eps_step_denominator
        self.eps_end = 0.1
        self.burn_in = 5
        self.net_copy_interval = 50

    def finalize(self):
        pass

    def take_snapshot(self):
        self.old_net.load_state_dict(self.net.state_dict())

    def make_actions(self, t, **kwargs):
        greedy = kwargs['greedy'] if 'greedy' in kwargs else True
        if greedy:
            return self.do_greedy_actions(t)
        else:
            if t % 2 == 0:
                self.eps = self.eps_end + max(0., (self.eps_start - self.eps_end)
                                              * (self.eps_step - max(0., self.step)) / self.eps_step)
                if self.local_random.random() < self.eps:
                    exploration_actions_t0, exploration_actions_t1 = self.environment.exploratory_actions(self.agent_exploration_policy)
                    self.next_exploration_actions = exploration_actions_t1
                    return exploration_actions_t0
                else:
                    greedy_acts = self.do_greedy_actions(t)
                    self.next_exploration_actions = None
                    return greedy_acts
            else:
                if self.next_exploration_actions is not None:
                    return self.next_exploration_actions
                else:
                    return self.do_greedy_actions(t)

    def do_greedy_actions(self, time_t):
        cur_state = self.environment.get_state_ref()
        actions, _, _ = self.net(time_t % 2, cur_state, None, greedy_acts=True)
        actions = list(actions.cpu().numpy())
        return actions

    def agent_exploration_policy(self, i):
        return self.pick_random_actions(i)

    def run_simulation(self):
        selected_idx = self.advance_pos_and_sample_indices()
        self.environment.setup([self.train_g_list[idx] for idx in selected_idx],
                           [self.train_initial_obj_values[idx] for idx in selected_idx],
                           training=True)
        self.post_env_setup()

        final_st = [None] * len(selected_idx)
        final_acts = np.empty(len(selected_idx), dtype=np.int); final_acts.fill(-1)

        t = 0
        while not self.environment.is_terminal():
            list_at = self.make_actions(t, greedy=False)

            non_exhausted_before, = np.where(~self.environment.exhausted_budgets)
            list_st = self.environment.clone_state(non_exhausted_before)
            self.environment.step(list_at)

            non_exhausted_after, = np.where(~self.environment.exhausted_budgets)
            exhausted_after, = np.where(self.environment.exhausted_budgets)

            nonterm_indices = np.flatnonzero(np.isin(non_exhausted_before, non_exhausted_after))
            nonterm_st = [list_st[i] for i in nonterm_indices]
            nonterm_at = [list_at[i] for i in non_exhausted_after]
            rewards = np.zeros(len(nonterm_at), dtype=np.float)
            nonterm_s_prime = self.environment.clone_state(non_exhausted_after)

            now_term_indices = np.flatnonzero(np.isin(non_exhausted_before, exhausted_after))
            term_st = [list_st[i] for i in now_term_indices]
            for i in range(len(term_st)):
                g_list_index = non_exhausted_before[now_term_indices[i]]

                final_st[g_list_index] = term_st[i]
                final_acts[g_list_index] = list_at[g_list_index]

            if len(nonterm_at) > 0:
                self.mem_pool.add_list(nonterm_st, nonterm_at, rewards, nonterm_s_prime, [False] * len(nonterm_at), t % 2)

            t += 1

        final_at = list(final_acts)
        rewards = self.environment.rewards
        final_s_prime = None
        self.mem_pool.add_list(final_st, final_at, rewards, final_s_prime, [True] * len(final_at), (t - 1) % 2)

    def post_env_setup(self):
        pass

    def get_default_hyperparameters(self):
        hyperparams = {'learning_rate': 0.0001,
                       'epsilon_start': 1,
                       'mem_pool_to_steps_ratio': 1,
                       'latent_dim': 64,
                       'hidden': 32,
                       'embedding_method': 'mean_field',
                       'max_lv': 5,
                       'eps_step_denominator': 10}
        return hyperparams

