import torch
from ExpReplay.ErDist import ErDist
from .Metrics_pytorch import WideNormMetric


class MlpDistEncoder(torch.nn.Module):
    """
    Just a NN model to learn the minimum action distance embedding
    """

    def __init__(self, in_d: int, out_d: int, dist_type: str = "WideNorm", in_dist_d: int = 32, out_dist_d: int = 32):
        super(MlpDistEncoder, self).__init__()
        ## LECUN NORMAL INIT IS THE DEFAULT
        self.l1 = torch.nn.Linear(in_d, 512, bias=False)
        self.l2 = torch.nn.Linear(512, 256, bias=False)
        self.l3 = torch.nn.Linear(256, 128, bias=False)
        self.l4 = torch.nn.Linear(128, out_d, bias=False)

        torch.nn.init.kaiming_normal_(self.l1.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l2.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l3.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l4.weight.data, mode='fan_in', nonlinearity='linear')

        self.activation = torch.nn.SELU()
        assert dist_type in ["WideNorm", "L1"], "The distance type must be WideNorm or L1"
        self.dist_type = dist_type
        if self.dist_type == "WideNorm":
            self.WideNorm_dist = WideNormMetric(out_d, in_dist_d, out_dist_d, symmetric=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        x = self.activation(x)
        x = self.l4(x)
        return x

    def wide_norm(self, z1: torch.Tensor, z2: torch.Tensor):
        return self.WideNorm_dist(z1, z2)

    def l1_norm(self, z1: torch.Tensor, z2: torch.Tensor):
        return torch.norm((z1 - z2), p=1, dim=1)

    def dist(self, z1, z2):
        if self.dist_type == "WideNorm":
            return self.wide_norm(z1, z2)
        elif self.dist_type == "L1":
            return self.l1_norm(z1, z2)
    
    def training_step(self, experience_replay: ErDist, optimizer: torch.optim.Optimizer, config: dict):
        """
        A training step for the model
        :param experience_replay: the experience replay to sample from
        :param optimizer: the optimizer to use
        :param config: the configuration of the training
        :return:
            loss_o: the loss for the objective
            loss_c: the loss for the constrains
            loss: loss_o + w * loss_c
        """

        self.train()

        # for _ in trange(self.config["gradient_steps"], desc='Step of gradient', leave=True):
        s1_o, s2_o, d_traj_o = experience_replay.get_batch(batch_size=config["batch_size_o"],
                                                          d_tresh=config["max_dist_obj"])
        z1_o = self(s1_o)
        z2_o = self(s2_o)
        # TODO check whether is better to use the log or the discounted version
        #loss_o = ((torch.log(self.model.dist(z1_o, z2_o) + 1) - torch.log(d_traj_o + 1)).pow(2)).mean()
        loss_o = ((config["dist_discount"] ** d_traj_o) * (self.dist(z1_o, z2_o) - d_traj_o).pow(2)).mean()

        s1_c, s2_c, d_traj_c = experience_replay.get_batch(batch_size=config["batch_size_c"],
                                                          d_tresh=config["max_dist_con"])
        z1_c = self(s1_c)
        z2_c = self(s2_c)
        loss_c = ((config["dist_discount"] ** d_traj_c) * (torch.max(self.dist(z1_c, z2_c) - d_traj_c, torch.zeros_like(d_traj_c)).pow(2))).sum()
        #loss_c = (torch.max(torch.log(self.model.dist(z1_c, z2_c) + 1) - torch.log(d_traj_c + 1), torch.zeros_like(d_traj_c)).pow(2)).sum()

        loss = loss_o + config["weight_constrains"] * loss_c

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss_o.detach(), loss_c.detach(), loss.detach()


class ActionEncoder(torch.nn.Module):
    """
    Just a NN model to learn the action embedding
    """
    def __init__(self, in_a_d, out_d):
        super(ActionEncoder, self).__init__()
        self.l1_a = torch.nn.Linear(in_a_d, 512, bias=True)
        self.l2_a = torch.nn.Linear(512, 256, bias=True)
        self.l3_a = torch.nn.Linear(256, 128, bias=True)
        self.l4_a = torch.nn.Linear(128, out_d, bias=True)

        self.l1_za = torch.nn.Linear(out_d*2, 1024, bias=True)
        self.l2_za = torch.nn.Linear(1024, 512, bias=True)
        self.l3_za = torch.nn.Linear(512, 256, bias=True)
        self.l4_za = torch.nn.Linear(256, 128, bias=True)
        self.l5_za = torch.nn.Linear(128, out_d, bias=True)

        torch.nn.init.kaiming_normal_(self.l1_a.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l2_a.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l3_a.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l4_a.weight.data, mode='fan_in', nonlinearity='linear')

        torch.nn.init.kaiming_normal_(self.l1_za.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l2_za.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l3_za.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l4_za.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l5_za.weight.data, mode='fan_in', nonlinearity='linear')

        self.activation = torch.nn.SELU()

    def forward(self, z, a):
        a = self.l1_a(a)
        a = self.activation(a)
        a = self.l2_a(a)
        a = self.activation(a)
        a = self.l3_a(a)
        a = self.activation(a)
        a = self.l4_a(a)

        za = torch.cat((z, a), dim=1)
        za = self.l1_za(za)
        za = self.activation(za)
        za = self.l2_za(za)
        za = self.activation(za)
        za = self.l3_za(za)
        za = self.activation(za)
        za = self.l4_za(za)
        za = self.activation(za)
        za = self.l5_za(za)
        return za

    def training_step(self, encoder: torch.nn.Module, experience_replay: ErDist, optimizer: torch.optim.Optimizer,
                      config: dict):

        """
        A training step for the model
        Args:
            encoder: the state encoder to use
            experience_replay: the experience replay to sample from
            optimizer: the optimizer to use
            config: the configuration of the training

        Returns:
            loss: the loss of the training step
        """

        self.train()
        encoder.eval()

        s1, s2, a = experience_replay.get_batch_action(batch_size=config["batch_size_action"])

        z1 = encoder(s1)
        z2 = encoder(s2)

        za = self(z1, a)

        loss = (z2 - za).pow(2).mean()
        #loss = ((encoder.dist(z2, za) + encoder.dist(za, z2)).pow(2)).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.detach()


class LearnedModel(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, action_encoder: torch.nn.Module):
        super(LearnedModel, self).__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.action_encoder = action_encoder
        self.action_encoder.eval()

    def forward(self, x1, x2):
        with torch.no_grad():
            return self.encode_state_dist(x1, x2)

    def encode_state(self, x):
        with torch.no_grad():
            return self.encoder(x)

    def encode_action(self, z, a):
        with torch.no_grad():
            return self.action_encoder(z, a)

    def eval_encode_state(self, x):
        with torch.no_grad():
            x = torch.FloatTensor(x)
            z = self.encode_state(x)
            return z

    def eval_encode_state_dist(self, x1, x2):
        with torch.no_grad():
            x1 = torch.FloatTensor(x1)
            x2 = torch.FloatTensor(x2)
            z1 = self.encode_state(x1)
            z2 = self.encode_state(x2)
            dist = self.encoder.dist(z1, z2)
        return dist

    def eval_encode_state_action_dist(self, x1, x1_a, x2, x2_a):
        with torch.no_grad():
            x1 = torch.FloatTensor(x1)
            x1_a = torch.FloatTensor(x1_a)
            x2 = torch.FloatTensor(x2)
            x2_a = torch.FloatTensor(x2_a)
            z1 = self.encode_state(x1) + self.encode_action(self.encode_state(x1).detach(), x1_a)
            z2 = self.encode_state(x2) + self.encode_action(self.encode_state(x2).detach(), x2_a)
            dist = self.encoder.dist(z1, z2)
        return dist

    def eval_encode_next_state(self, x1, x1_a):
        with torch.no_grad():
            x1 = torch.FloatTensor(x1)
            a1 = torch.FloatTensor(x1_a)
            z1 = self.encode_state(x1)
            z2 = self.encode_action(z1, a1)
        return z2

    def eval_encode_state_next_state(self, x1, x2, x1_a):
        with torch.no_grad():
            x1 = torch.FloatTensor(x1)
            x2 = torch.FloatTensor(x2)
            a1 = torch.FloatTensor(x1_a)
            z2 = self.encode_state(x2).detach()
            z_a_1 = self.encode_next_state(x1, a1)
        return z_a_1, z2

    def eval_encode_state_next_state_dist(self, x1, x2, x1_a):
        with torch.no_grad():
            x1 = torch.FloatTensor(x1)
            x2 = torch.FloatTensor(x2)
            a1 = torch.FloatTensor(x1_a)
            z2 = self.encode_state(x2).detach()
            z_a_1 = self.encode_next_state(x1, a1)
            d_out = self.encoder.dist(z2, z_a_1)
        return d_out


