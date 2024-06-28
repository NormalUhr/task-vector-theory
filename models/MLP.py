import torch.nn as nn
from backpack import extend
from models.model_utils import ptv, vtp


class MLP(nn.Module):
    def __init__(self, env_num=1, grayscale_model=False, hidden_dim=390, use_extend=False, use_color=False):
        super(MLP, self).__init__()
        self.channel = 4 if use_color else 2
        self.grayscale_model = grayscale_model
        if self.grayscale_model:
            lin1 = nn.Linear(14 * 14, hidden_dim)
        else:
            lin1 = nn.Linear(self.channel * 14 * 14, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin_list = []
        self.omega_list = []
        self.env_num = env_num
        for env in range(env_num):
            omega = nn.Linear(hidden_dim, 2)
            if use_extend:
                omega = extend(omega)
            lin_list.append(omega)
        # lin3 = nn.Linear(390, 1)
        for lin in [lin1, lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for lin in lin_list:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.omega_list.append(nn.Sequential(lin))
        self.phi = nn.Sequential(lin1, nn.ReLU(inplace=False), lin2, nn.ReLU(inplace=False))

    def load_device(self, device):
        self.phi = self.phi.to(device)
        for omega in self.omega_list:
            omega = omega.to(device)

    def forward(self, input, env_num=0):
        if self.grayscale_model:
            out = input.view(input.shape[0], self.channel, 14 * 14).sum(dim=1)
        else:
            out = input.view(input.shape[0], self.channel * 14 * 14)
        out = self.phi(out)
        if env_num < 0:
            return out
        out = self.omega_list[env_num](out)
        return out

    def get_omega_reg(self, requires_grad=True):
        if requires_grad:
            for omega in self.omega_list:
                omega.requires_grad_(True)
        res = 0
        if self.env_num < 1:
            return res
        for i in range(self.env_num - 1):
            res += (ptv(self.omega_list[i].parameters()) - ptv(self.omega_list[i + 1].parameters())).norm(p=2)
        return res ** 0.5

    def average_omega(self):
        sum_param = None
        for omega in self.omega_list:
            if sum_param is not None:
                sum_param += ptv(omega.parameters())
            else:
                sum_param = ptv(omega.parameters())
        avg_param = sum_param / len(self.omega_list)

        for omega in self.omega_list:
            vtp(avg_param, omega.parameters())

    def clear_phi_grad(self):
        for param in self.phi.parameters():
            param.grad = None
            param.requires_grad_(True)
        # self.phi.requires_grad_(True)

    def clear_omega_grad(self):
        for omega in self.omega_list:
            for param in omega.parameters():
                param.grad = None
                param.requires_grad_(True)

    def turn_on_phi_grad(self):
        for param in self.phi:
            param.requires_grad_(True)

    def shut_down_phi_grad(self):
        self.phi.requires_grad_(False)

    def shut_down_all_omega_grad(self):
        for omega in self.omega_list:
            for param in omega.parameters():
                param.grad = None
                param.requires_grad_(True)

    def shut_down_omega_grad(self, env_num):
        omega = self.omega_list[env_num]
        for param in omega.parameters():
            param.grad = None
            param.requires_grad_(True)


def MLP360(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=360, use_color=use_color)


def MLP390(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=390, use_color=use_color)


def MLP720(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=720, use_color=use_color)


def MLP180(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=180, use_color=use_color)


def MLP540(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=540, use_color=use_color)


def MLP1k(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=1024, use_color=use_color)


def MLP2k(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=2048, use_color=use_color)


def MLP_Gray(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=360, grayscale_model=True, use_color=use_color)


def MLP_Extend(env_num=1, use_color=False):
    return MLP(env_num=env_num, hidden_dim=360, use_extend=True, use_color=use_color)
