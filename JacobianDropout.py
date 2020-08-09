import torch
from torch import nn

class JacobianDropout(nn.Module):
    def __init__(self, p, beta):
        super().__init__()
        self.p = p
        self.jacobian = None
        self.beta = beta

    def update_jacobian(self, batch_jacobian):
        if self.jacobian is not None:
            self.jacobian = self.beta * self.jacobian + (1 - self.beta) * batch_jacobian.sum(dim=1)
        else:
            self.jacobian = batch_jacobian
            
    def forward(self, x, batch_jacobian=None):
        if batch_jacobian is not None:
            self.update_jacobian(batch_jacobian)
        
        if self.jacobian is None:
            j = torch.abs(-2 * torch.randn(size=(1, *x.shape[1:])) + 1)
            j.requires_grad = False
        else:
            j = torch.abs(self.jacobian)
            j.requires_grad = False
        #print(j.shape, x.shape)
        if self.train:
            numerator = j - j.min(dim=1).values
            divisor = j.max(dim=1).values - j.min(dim=1).values

            j_norm = torch.div(numerator, divisor)
            ones = torch.ones(x.shape)
            zeros = torch.zeros(x.shape)

            if x.is_cuda:
                j_norm = j_norm.cuda()
                ones = ones.cuda()
                zeros = zeros.cuda()
            
            alpha = self.p# / j_norm.mean()
            
            # idx = torch.where(j_norm > self.p, ones, zeros)
            idx = torch.bernoulli(1 - torch.clamp(alpha * j_norm, min=0, max=0.9999))
            return_tensor = x * idx
            return return_tensor
        else:
            return x
def hook_fn(m, i, o):
    # print(i)
    for grad in i:
        if grad is not None:
            m.update_jacobian(grad.mean(dim=0))

if __name__ == "__main__":
    teacher = nn.Sequential(
        nn.Linear(786, 10),
        nn.Softmax(dim=1)
    )

    net = nn.Sequential(
        nn.Linear(786, 256),
        JacobianDropout(0.3, 0.99),
        nn.LeakyReLU(),
        nn.Linear(256, 64),
        JacobianDropout(0.3, 0.99),
        nn.LeakyReLU(),
        nn.Linear(64, 10),
    )

    for m in net.modules():
        if isinstance(m, JacobianDropout):
            m.register_backward_hook(hook_fn)

    optim = torch.optim.Adam(net.parameters())
    loss_fn = nn.MSELoss()

    optim.zero_grad()

    inp = torch.randn((32, 786))

    out = net(inp)

    loss = loss_fn(out, teacher(inp))
    loss.backward()

    optim.step()



    # out = inp
    # for m in net.children():
    #     if isinstance(m, JacobianDropout):
    #         pre_dropout_out = out
    #     out = m(out)
            
            
    # #loss = loss_fn(out, torch.ones(out.shape))
    # out.backward(pre_dropout_out)

    # batch_jacobian = pre_dropout_out.grad
    # print(batch_jacobian)