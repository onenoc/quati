import torch
import torch.nn as nn
import math
# torch.autograd.set_detect_anomaly(True)


class ContinuousSparsemaxFunction(torch.autograd.Function):
    @classmethod
    def _integrate_phi_times_psi(cls, ctx, a, b):
        """Compute integral int_a^b phi(t) * psi(t).T."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        V = torch.zeros(
            (a.shape[0], 2, total_basis), dtype=ctx.dtype, device=ctx.device
        )
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            V[:, 0, start:offsets[j]] = basis_functions.integrate_t_times_psi(a, b)
            V[:, 1, start:offsets[j]] = basis_functions.integrate_t2_times_psi(a, b)
            start = offsets[j]
        return V

    @classmethod
    def _integrate_psi(cls, ctx, a, b):
        """Compute integral int_a^b psi(t)."""
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        v = torch.zeros(
            a.shape[0], total_basis, dtype=ctx.dtype, device=ctx.device
        )
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            v[:, start:offsets[j]] = basis_functions.integrate_psi(a, b)
            start = offsets[j]
        return v

    @classmethod
    def _integrate_phi(cls, ctx, a, b):
        """Compute integral int_a^b phi(t)."""
        v = torch.zeros(a.shape[0], 2, dtype=ctx.dtype, device=ctx.device)
        v[:, 0] = ((b ** 2 - a ** 2) / 2).squeeze(1)
        v[:, 1] = ((b ** 3 - a ** 3) / 3).squeeze(1)
        return v

    @classmethod
    def forward(cls, ctx, theta, psi):
        # We assume a truncated parabola.
        # We have:
        # theta = [mu/sigma**2, -1/(2*sigma**2)],
        # phi(t) = [t, t**2],
        # p(t) = [theta.dot(phi(t)) - A]_+,
        # supported on [mu - a, mu + a].
        ctx.dtype = theta.dtype
        ctx.device = theta.device
        ctx.psi = psi
        sigma = torch.sqrt(-0.5 / theta[:, 1])
        mu = theta[:, 0] * sigma ** 2
        A = -0.5 * (3.0 / (2 * sigma)) ** (2.0 / 3)
        a = torch.sqrt(-2 * A) * sigma
        A += mu ** 2 / (2 * sigma ** 2)
        left = (mu - a).unsqueeze(1)
        right = (mu + a).unsqueeze(1)
        V = cls._integrate_phi_times_psi(ctx, left, right)
        u = cls._integrate_psi(ctx, left, right)
        #r: bs x nb_basis
        r = torch.matmul(theta.unsqueeze(1), V).squeeze(1) - A.unsqueeze(1) * u
        ctx.save_for_backward(mu, a, V, u)
        return r

    @classmethod
    def backward(cls, ctx, grad_output):
        mu, a, V, u = ctx.saved_tensors
        # J.T = int_{-a}^{+a} phi(t+mu)*psi(t+mu).T
        # - (int_{-a}^{+a} phi(t+mu)) * (int_{-a}^{+a} psi(t+mu).T) / (2*a)
        left = (mu - a).unsqueeze(1)
        right = (mu + a).unsqueeze(1)
        i_phi = cls._integrate_phi(ctx, left, right)
        ger = torch.bmm(i_phi.unsqueeze(2), u.unsqueeze(1))
        # ger = torch.einsum('bi,bj->bij', (i_phi, u))
        J = V - ger / (2 * a.unsqueeze(1).unsqueeze(2))
        grad_input = torch.matmul(J, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input, None


class ContinuousSparsemax(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousSparsemax, self).__init__()
        self.psi = psi
        self.plus = torch.nn.ReLU()

    def _phi(self,t):
        return 1.0/math.sqrt(2*math.pi)*torch.exp(-0.5*t**2)

    def truncated_parabola(self,t,mu,sigma_sq):
        return self.plus(-(t-mu)**2/(2*sigma_sq)+0.5*(3/(2*torch.sqrt(sigma_sq)))**(2./3.))

    def gaussian_rbf(self,t,bandwidth):
        return torch.exp(-0.5*t**2/bandwidth)

    def beta_exp(self,t):
        return self.plus(1+t)

    def forward(self, theta):
        #size: 1 x nb_basis x 1
        mu_basis = self.psi[0].mu.unsqueeze(-1)
        sigma_basis = self.psi[0].sigma.unsqueeze(-1)
        a = -5
        b = 5
        dx = 100
        bandwidth = 0.01
        #size: 1 x 1 x dx
        T = torch.linspace(a,b,dx,device=theta.device).unsqueeze(0).unsqueeze(0)
        inducing_locations = torch.linspace(0,1,theta.shape[1]-2,device=theta.device)

        K_inputs = torch.cdist(inducing_locations.unsqueeze(-1),torch.linspace(a,b,dx,device=theta.device).unsqueeze(-1))
        #size: inducing x dx
        K = self.gaussian_rbf(K_inputs,bandwidth)
        alpha = theta[:,2:]
        #size: bx x 1 x dx
        f = torch.matmul(alpha,K).unsqueeze(1)
        #mu, sigma: bs x 1 x 1
        sigma = torch.sqrt(-0.5/theta[:,1])
        sigma_sq = (-0.5/theta[:,1]).unsqueeze(-1).unsqueeze(-1)
        mu = (theta[:,0]*sigma**2).unsqueeze(-1).unsqueeze(-1)

        phi1_upper = mu_basis-T
        phi1_lower = sigma_basis
        phi1 = self._phi(phi1_upper/phi1_lower)/phi1_lower

        exp_terms = self.beta_exp(f-0.5*(mu-T)**2/(sigma_sq)).clamp(min=1e-8)

        #size: bs x 1 x dx
        #unnormalized_density = self.truncated_parabola(T,mu,sigma_sq)
        #size: bs x 1 x 1
        #Z = torch.trapz(unnormalizedi_density,torch.linspace(a,b,dx,device=theta.device),dim=-1).unsqueeze(-1)
        Z = torch.trapz(exp_terms,torch.linspace(a,b,dx,device=theta.device),dim=-1).unsqueeze(-1)
        numerical_integral = torch.trapz(phi1*exp_terms/Z,torch.linspace(a,b,dx,device=theta.device),dim=-1)
        return numerical_integral



    #def forward(self, theta):
    #    return ContinuousSparsemaxFunction.apply(theta, self.psi)
