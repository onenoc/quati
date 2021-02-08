import torch
import math
import numpy as np


class BasisFunctions(object):
    def __len__(self):
        """Number of basis functions."""
        raise NotImplementedError

    def evaluate(self, t):
        raise NotImplementedError

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        raise NotImplementedError

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        raise NotImplementedError

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        raise NotImplementedError


class PowerBasisFunctions(BasisFunctions):
    """Function phi(t) = t**degree."""

    def __init__(self, degree):
        self.degree = degree.unsqueeze(0)

    def __len__(self):
        """Number of basis functions."""
        return self.degree.size(1)

    def evaluate(self, t):
        return t ** self.degree

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        return (b ** (self.degree + 3) - a ** (self.degree + 3)) / (
            self.degree + 3
        )

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        return (b ** (self.degree + 2) - a ** (self.degree + 2)) / (
            self.degree + 2
        )

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        return (b ** (self.degree + 1) - a ** (self.degree + 1)) / (
            self.degree + 1
        )


class SineBasisFunctions(BasisFunctions):
    """Function phi(t) = sin(omega*t)."""

    def __init__(self, omega):
        self.omega = omega.unsqueeze(0)

    def __repr__(self):
        return f"SineBasisFunction(omega={self.omega})"

    def __len__(self):
        """Number of basis functions."""
        return self.omega.size(1)

    def evaluate(self, t):
        return torch.sin(self.omega * t)

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        # The antiderivative of (t**2)*sin(omega*t) is
        # ((2-(t**2)*(omega**2))*cos(omega*t) + 2*omega*t*sin(omega*t)) / omega**3.  # noqa
        return (
            (2 - (b ** 2) * (self.omega ** 2)) * torch.cos(self.omega * b)
            + 2 * self.omega * b * torch.sin(self.omega * b)
            - (2 - (a ** 2) * (self.omega ** 2)) * torch.cos(self.omega * a)
            - 2 * self.omega * a * torch.sin(self.omega * a)
        ) / (self.omega ** 3)

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        # The antiderivative of t*sin(omega*t) is
        # (sin(omega*t) - omega*t*cos(omega*t)) / omega**2.
        return (
            torch.sin(self.omega * b)
            - self.omega * b * torch.cos(self.omega * b)
            - torch.sin(self.omega * a)
            + self.omega * a * torch.cos(self.omega * a)
        ) / (self.omega ** 2)

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        # The antiderivative of sin(omega*t) is -cos(omega*t)/omega.
        return (
            -torch.cos(self.omega * b) + torch.cos(self.omega * a)
        ) / self.omega


class CosineBasisFunctions(BasisFunctions):
    """Function phi(t) = cos(omega*t)."""

    def __init__(self, omega):
        self.omega = omega.unsqueeze(0)

    def __repr__(self):
        return f"CosineBasisFunction(omega={self.omega})"

    def __len__(self):
        """Number of basis functions."""
        return self.omega.size(1)

    def evaluate(self, t):
        return torch.cos(self.omega * t)

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        # The antiderivative of (t**2)*cos(omega*t) is
        # (((t**2)*(omega**2)-2)*cos(omega*t) + 2*omega*t*sin(omega*t)) / omega**3.  # noqa
        return (
            ((b ** 2) * (self.omega ** 2) - 2) * torch.sin(self.omega * b)
            + 2 * self.omega * b * torch.cos(self.omega * b)
            - ((a ** 2) * (self.omega ** 2) - 2) * torch.sin(self.omega * a)
            - 2 * self.omega * a * torch.cos(self.omega * a)
        ) / (self.omega ** 3)

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        # The antiderivative of t*cos(omega*t) is
        # (cos(omega*t) + omega*t*sin(omega*t)) / omega**2.
        return (
            torch.cos(self.omega * b)
            + self.omega * b * torch.sin(self.omega * b)
            - torch.cos(self.omega * a)
            - self.omega * a * torch.sin(self.omega * a)
        ) / (self.omega ** 2)

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        # The antiderivative of cos(omega*t) is sin(omega*t)/omega.
        return (
            torch.sin(self.omega * b) - torch.sin(self.omega * a)
        ) / self.omega


class GaussianBasisFunctions(BasisFunctions):
    """Function phi(t) = Gaussian(t; mu, sigma_sq)."""

    def __init__(self, mu, sigma):
        self.mu = mu.unsqueeze(0)
        self.sigma = sigma.unsqueeze(0)

    def __repr__(self):
        return f"GaussianBasisFunction(mu={self.mu}, sigma={self.sigma})"

    def __len__(self):
        """Number of basis functions."""
        return self.mu.size(1)

    def _phi(self, t):
        return 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * t ** 2)

    def _Phi(self, t):
        return 0.5 * (1 + torch.erf(t / math.sqrt(2)))

    def _integrate_product_of_gaussians(self, mu, sigma_sq):
        sigma = torch.sqrt(self.sigma ** 2 + sigma_sq)
        return self._phi((mu - self.mu) / sigma) / sigma

    def evaluate(self, t):
        return self._phi((t - self.mu) / self.sigma) / self.sigma

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        return (
            (self.mu ** 2 + self.sigma ** 2)
            * (
                self._Phi((b - self.mu) / self.sigma)
                - self._Phi((a - self.mu) / self.sigma)
            )
            - (
                self.sigma
                * (b + self.mu)
                * self._phi((b - self.mu) / self.sigma)
            )
            + (
                self.sigma
                * (a + self.mu)
                * self._phi((a - self.mu) / self.sigma)
            )
        )

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        return self.mu * (
            self._Phi((b - self.mu) / self.sigma)
            - self._Phi((a - self.mu) / self.sigma)
        ) - self.sigma * (
            self._phi((b - self.mu) / self.sigma)
            - self._phi((a - self.mu) / self.sigma)
        )

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        return self._Phi((b - self.mu) / self.sigma) - self._Phi(
            (a - self.mu) / self.sigma
        )

    def integrate_t2_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t**2 * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (self.mu * sigma_sq + mu * self.sigma ** 2) / (
            self.sigma ** 2 + sigma_sq
        )
        sigma_sq_tilde = ((self.sigma ** 2) * sigma_sq) / (
            self.sigma ** 2 + sigma_sq
        )
        return S_tilde * (mu_tilde ** 2 + sigma_sq_tilde)

    def integrate_t_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (self.mu * sigma_sq + mu * self.sigma ** 2) / (
            self.sigma ** 2 + sigma_sq
        )
        return S_tilde * mu_tilde

    def integrate_psi_kernel_exp(self,mu,sigma_sq,alpha):
        a = -5
        b = 5
        bandwidth=.01
        dt_num = 1000
        t_raw = torch.linspace(a,b,dt_num,device=mu.device)
        t = t_raw.unsqueeze(0).unsqueeze(0)
        mu = mu.unsqueeze(-1)
        sigma_sq = sigma_sq.unsqueeze(-1)
        mu_basis = self.mu.unsqueeze(-1)
        sigma_basis = self.sigma.unsqueeze(-1)
        y1 = 1/(torch.sqrt(2*math.pi*sigma_sq))*torch.exp(-(mu-t).pow(2)/(2*sigma_sq))
        y2 = 1/(math.sqrt(2*math.pi)*sigma_basis)*torch.exp(-(mu_basis-t).pow(2)/(2*sigma_basis.pow(2)))
        dist_matrix = torch.cdist(torch.linspace(0,1,alpha.shape[1],device=mu.device).unsqueeze(-1),torch.linspace(a,b,dt_num,device=mu.device).unsqueeze(-1))
        K_matrix=torch.exp(-dist_matrix.pow(2)/(2*bandwidth))
        y3 = torch.exp(torch.mm(alpha,K_matrix))
        y3 = y3.unsqueeze(1)
        #y3 = torch.ones(y3.shape,device=y3.device)
        Z = torch.trapz(y1*y3,torch.linspace(a,b,dt_num,device=mu.device),dim=-1).unsqueeze(-1)
        y = y1*y2*y3
        y=y/Z
        #Need normalization constant
        integral = torch.trapz(y,torch.linspace(a,b,dt_num,device=mu.device),dim=-1)
        return integral

    def integrate_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * psi(t)."""
        return self._integrate_product_of_gaussians(mu, sigma_sq)
