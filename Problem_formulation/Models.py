import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_d, out_d):
        super(Encoder, self).__init__()
        ## LECUN NORMAL INIT IS THE DEFAULT
        self.l1 = nn.Linear(in_d, 128, bias=False)
        self.l2 = nn.Linear(128, 128, bias=False)
        self.l3 = nn.Linear(128, out_d, bias=False)

        torch.nn.init.kaiming_normal_(self.l1.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l2.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l3.weight.data, mode='fan_in', nonlinearity='linear')

        self.activation = torch.nn.SELU()

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        return x

    def dist(self, z1, z2):
        dist = torch.norm((z1 - z2), p=1, dim=1)
        return dist


class ActionEncoder(nn.Module):
    def __init__(self, in_a_d, out_d):
        super(ActionEncoder, self).__init__()
        self.l1_a = nn.Linear(in_a_d, 128, bias=True)
        self.l2_a = nn.Linear(128, 128, bias=True)
        self.l3_a = nn.Linear(128, out_d, bias=True)

        self.l1_z = nn.Linear(out_d, 128, bias=True)
        self.l2_z = nn.Linear(128, 128, bias=True)
        self.l3_z = nn.Linear(128, out_d, bias=True)

        self.l1_za = nn.Linear(out_d*2, 128, bias=True)
        self.l2_za = nn.Linear(128, out_d, bias=True)

        torch.nn.init.kaiming_normal_(self.l1_a.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l2_a.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l3_a.weight.data, mode='fan_in', nonlinearity='linear')

        torch.nn.init.kaiming_normal_(self.l1_z.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l2_z.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l3_z.weight.data, mode='fan_in', nonlinearity='linear')

        torch.nn.init.kaiming_normal_(self.l1_za.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l2_za.weight.data, mode='fan_in', nonlinearity='linear')

        self.activation = torch.nn.SELU()


    def forward(self, z, a):
        a = self.l1_a(a)
        a = self.activation(a)
        a = self.l2_a(a)
        a = self.activation(a)
        a = self.l3_a(a)

        z = self.l1_z(z)
        z = self.activation(z)
        z = self.l2_z(z)
        z = self.activation(z)
        z = self.l3_z(z)

        za = torch.cat((a, z), 1)
        za = self.l1_za(za)
        za = self.activation(za)
        za = self.l2_za(za)
        return za


class LearnedModel(nn.Module):
    def __init__(self, in_d, out_d, in_a_d):
        super(LearnedModel, self).__init__()
        self.encoder = Encoder(in_d, out_d)
        self.action_encoder = ActionEncoder(in_a_d, out_d)

    def forward(self, x1, x2):
        return self.encode_state_dist(x1, x2)

    def encode_state(self, x):
        return self.encoder(x)

    def encode_state_dist(self, x1, x2):
        x1 = torch.FloatTensor(x1)
        x2 = torch.FloatTensor(x2)
        z1 = self.encode_state(x1)
        z2 = self.encode_state(x2)
        dist = self.encoder.dist(z1, z2)
        return dist

    def encode_state_action_dist(self, x1, x1_a, x2, x2_a):
        x1 = torch.FloatTensor(x1)
        x1_a = torch.FloatTensor(x1_a)
        x2 = torch.FloatTensor(x2)
        x2_a = torch.FloatTensor(x2_a)
        z1 = self.encode_state(x1) + self.encode_action(self.encode_state(x1).detach(), x1_a)
        z2 = self.encode_state(x2) + self.encode_action(self.encode_state(x2).detach(), x2_a)
        dist = self.encoder.dist(z1, z2)
        return dist

    def encode_next_state(self, x1, x1_a):
        x1 = torch.FloatTensor(x1)
        a1 = torch.FloatTensor(x1_a)
        z1 = self.encode_state(x1).detach()
        z_a = z1 + self.encode_action(z1, a1)
        return z_a

    def encode_state_next_state(self, x1, x2, x1_a):
        x1 = torch.FloatTensor(x1)
        x2 = torch.FloatTensor(x2)
        a1 = torch.FloatTensor(x1_a)
        z2 = self.encode_state(x2).detach()
        z_a_1 = self.encode_next_state(x1, a1)
        return z_a_1, z2

    def encode_state_next_state_dist(self, x1, x2, x1_a):
        x1 = torch.FloatTensor(x1)
        x2 = torch.FloatTensor(x2)
        a1 = torch.FloatTensor(x1_a)
        z2 = self.encode_state(x2).detach()
        z_a_1 = self.encode_next_state(x1, a1)
        d_out = self.encoder.dist(z2, z_a_1)
        return d_out

    def encode_action(self, z, a):
        return self.action_encoder(z, a)


class ImagesEncoder(nn.Module):
    def __init__(self, w, h, out_d, in_a_d):
        super(ImagesEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=1, stride=1, padding=0, dilation=1):
            return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=1), kernel_size=3, stride=1, padding=1), kernel_size=3, stride=1, padding=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=1), kernel_size=3, stride=1, padding=1), kernel_size=3, stride=1, padding=1)
        linear_input_size = convw * convh * 16
        self.head = nn.Linear(linear_input_size, 128)
        self.out = nn.Linear(128, out_d)

    def forward(self, x1, x2):
        x1 = torch.FloatTensor(x1)
        x2 = torch.FloatTensor(x2)

        z1 = self.encode(x1)
        z2 = self.encode(x2)
        dist = self.dist(z1, z2)
        return dist

    def encode(self, x):
        x = x.permute(0, 3, 1, 2)
        x = nn.functional.selu(self.conv1(x))
        x = nn.functional.selu(self.conv2(x))
        x = nn.functional.selu(self.conv3(x))
        x = nn.functional.selu(self.head(x.view(x.size(0), -1)))
        return self.out(x)

    def dist(self, z1, z2):
        dist = torch.abs(z1 - z2).sum(1)
        return dist

