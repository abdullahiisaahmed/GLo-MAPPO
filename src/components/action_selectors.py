import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        action_nvec = getattr(self.args, "action_nvec", None)
        if action_nvec is not None:
            parts, start = [], 0
            for n in action_nvec:
                probs = masked_policies[..., start:start + n]
                if test_mode and self.test_greedy:
                    parts.append(probs.max(dim=-1)[1].unsqueeze(-1))
                else:
                    parts.append(Categorical(probs).sample().long().unsqueeze(-1))
                start += n
            return th.cat(parts, dim=-1)
        else:
            if test_mode and self.test_greedy:
                return masked_policies.max(dim=2)[1]
            else:
                return Categorical(masked_policies).sample().long()


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        action_nvec = getattr(self.args, "action_nvec", None)
        if action_nvec is not None:
            # Per-dim epsilon-greedy for MultiDiscrete
            random_numbers = th.rand_like(agent_inputs[:, :, 0])
            pick_random = (random_numbers < self.epsilon).long()  # (B, A)
            parts, start = [], 0
            for n in action_nvec:
                q_dim = masked_q_values[..., start:start + n]
                greedy = q_dim.max(dim=-1)[1].unsqueeze(-1)          # (B, A, 1)
                avail_dim = avail_actions[..., start:start + n].float()
                random = Categorical(avail_dim).sample().long().unsqueeze(-1)  # (B, A, 1)
                parts.append(pick_random.unsqueeze(-1) * random + (1 - pick_random.unsqueeze(-1)) * greedy)
                start += n
            return th.cat(parts, dim=-1)
        else:
            random_numbers = th.rand_like(agent_inputs[:, :, 0])
            pick_random = (random_numbers < self.epsilon).long()
            random_actions = Categorical(avail_actions.float()).sample().long()
            return pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        action_nvec = getattr(self.args, "action_nvec", None)
        if action_nvec is not None:
            # Per-dim Categorical sampling for MultiDiscrete
            parts, start = [], 0
            for n in action_nvec:
                probs = agent_inputs[..., start:start + n]
                parts.append(Categorical(probs).sample().long().unsqueeze(-1))
                start += n
            return th.cat(parts, dim=-1)
        else:
            return Categorical(agent_inputs).sample().long()


REGISTRY["soft_policies"] = SoftPoliciesSelector