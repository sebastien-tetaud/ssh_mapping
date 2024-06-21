import torch

def gaspari_cohn(r, c):
    """
    Gaspari-Cohn function. Inspired from E.Cosmes.
        
    Args: 
        r : array of value whose the Gaspari-Cohn function will be applied
        c : Distance above which the return values are zeros

    Returns:  smoothed values 
    """ 
    if isinstance(r, (float, int)):
        ra = torch.tensor([r], dtype=torch.float64)
    else:
        ra = torch.tensor(r, dtype=torch.float64)
        
    if c <= 0:
        return torch.zeros_like(ra)
    else:
        ra = 2 * torch.abs(ra) / c
        gp = torch.zeros_like(ra)
        
        # Conditions for the Gaspari-Cohn function
        i1 = ra <= 1.
        i2 = (ra > 1.) & (ra <= 2.)
        
        # Applying the Gaspari-Cohn function
        gp[i1] = -0.25 * ra[i1]**5 + 0.5 * ra[i1]**4 + 0.625 * ra[i1]**3 - (5./3.) * ra[i1]**2 + 1.
        gp[i2] = (1./12.) * ra[i2]**5 - 0.5 * ra[i2]**4 + 0.625 * ra[i2]**3 + (5./3.) * ra[i2]**2 - 5. * ra[i2] + 4. - (2./3.) / ra[i2]
        
        if isinstance(r, float):
            gp = gp.item()
            
    return gp

def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j * torch.nn.functional.pad(x, (1, 1)), norm=norm)[...,1:x.shape[-1]+1]

def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1, -2), norm=norm).transpose(-1, -2)

def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    return dstI2D(dstI2D(f.double()) / operator_dst)


class Qgm:

    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self, dx=None, dy=None, dt=None, SSH=None, c=None, g=9.81, f=1e-4, Kdiffus=None, device='cpu'):

        # Grid shape
        ny, nx = dx.shape
        self.nx = nx
        self.ny = ny

        # Grid spacing
        dx = dy = (torch.nanmean(dx) + torch.nanmean(dy)) / 2
        self.dx = dx.double()
        self.dy = dy.double()

        # Time step
        self.dt = dt

        # Gravity
        self.g = torch.tensor(g).to(device=device)

        # Coriolis
        if hasattr(f, "__len__"):
            self.f = (torch.nanmean(torch.tensor(f)) * torch.ones((self.ny, self.nx))).double().to(device=device)
        else:
            self.f = (f * torch.ones((self.ny, self.nx))).double().to(device=device)

        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = (torch.nanmean(torch.tensor(c)) * torch.ones((self.ny, self.nx))).double().to(device=device)
        else:
            self.c = (c * torch.ones((self.ny, self.nx))).double().to(device=device)

        # Elliptical inversion operator
        x, y = torch.meshgrid(torch.arange(1, nx - 1, dtype=torch.float64),
                              torch.arange(1, ny - 1, dtype=torch.float64))
        x = x.to(device=device)
        y = y.to(device=device)
        laplace_dst = 2 * (torch.cos(torch.pi / (nx - 1) * x) - 1) / self.dx ** 2 + \
                      2 * (torch.cos(torch.pi / (ny - 1) * y) - 1) / self.dy ** 2
        self.helmoltz_dst = self.g / self.f.mean() * laplace_dst - self.g * self.f.mean() / self.c.mean() ** 2

        # get land pixels
        if SSH is not None:
            isNAN = torch.isnan(SSH).to(device=device)
        else:
            isNAN = None

        ################
        # Mask array
        ################

        # mask=3 away from the coasts
        mask = torch.zeros((ny, nx), dtype=torch.int64) + 3

        # mask=1 for borders of the domain 
        mask[0, :] = 1
        mask[:, 0] = 1
        mask[-1, :] = 1
        mask[:, -1] = 1

        # mask=2 for pixels adjacent to the borders 
        mask[1, 1:-1] = 2
        mask[1:-1, 1] = 2
        mask[-2, 1:-1] = 2
        mask[-3, 1:-1] = 2
        mask[1:-1, -2] = 2
        mask[1:-1, -3] = 2

        # mask=0 on land 
        if isNAN is not None:
            mask[isNAN] = 0
            indNan = torch.argwhere(isNAN)
            for i, j in indNan:
                for p1 in range(-2, 3):
                    for p2 in range(-2, 3):
                        itest = i + p1
                        jtest = j + p2
                        if ((itest >= 0) & (itest <= ny - 1) & (jtest >= 0) & (jtest <= nx - 1)):
                            # mask=1 for coast pixels
                            if (mask[itest, jtest] >= 2) and (p1 in [-1, 0, 1] and p2 in [-1, 0, 1]):
                                mask[itest, jtest] = 1
                            # mask=1 for pixels adjacent to the coast
                            elif (mask[itest, jtest] == 3):
                                mask[itest, jtest] = 2

        self.mask = mask.to(device=device)
        self.ind0 = (mask == 0).to(device=device)
        self.ind1 = (mask == 1).to(device=device)
        self.ind2 = (mask == 2).to(device=device)
        self.ind12 = (self.ind1 + self.ind2).to(device=device)

        # Diffusion coefficient 
        self.Kdiffus = Kdiffus

    def h2uv(self, h):
        """ SSH to U,V

        Args:
            h (2D array): SSH field.

        Returns:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
        """
    
        u = torch.zeros_like(h)
        v = torch.zeros_like(h)

        u[..., 1:-1, 1:] = - self.g / self.f[None, 1:-1, 1:] * (h[..., 2:, :-1] + h[..., 2:, 1:] - h[..., :-2, 1:] - h[..., :-2, :-1]) / (4 * self.dy)
        v[..., 1:, 1:-1] = self.g / self.f[None, 1:, 1:-1] * (h[..., 1:, 2:] + h[..., :-1, 2:] - h[..., :-1, :-2] - h[..., 1:, :-2]) / (4 * self.dx)
        
        u = torch.where(torch.isnan(u), torch.tensor(0.0), u)
        v = torch.where(torch.isnan(v), torch.tensor(0.0), v)
            
        return u, v

    def h2pv(self, h, hbc, c=None):
        """ SSH to Q

        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius

        Returns:
            q: Potential Vorticity field
        """
        
        if c is None:
            c = self.c

        q = torch.zeros_like(h)

        q[..., 1:-1, 1:-1] = (
            self.g / self.f[None, 1:-1, 1:-1] * 
            ((h[..., 2:, 1:-1] + h[..., :-2, 1:-1] - 2 * h[..., 1:-1, 1:-1]) / self.dy ** 2 +
             (h[..., 1:-1, 2:] + h[..., 1:-1, :-2] - 2 * h[..., 1:-1, 1:-1]) / self.dx ** 2) - 
            self.g * self.f[None, 1:-1, 1:-1] / (c[None, 1:-1, 1:-1] ** 2) * h[..., 1:-1, 1:-1]
        )

        q = torch.where(torch.isnan(q), torch.tensor(0.0), q)

        q[..., self.ind12] = - self.g * self.f[None,self.ind12] / (c[None,self.ind12] ** 2) * hbc[...,self.ind12]
        q[..., self.ind0] = 0

        return q

    def rhs(self, u, v, q0, way=1):
        """ increment

        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)

        Returns:
            rhs (2D array): advection increment
        """

        # Upwind current
        u_on_T = way * 0.5 * (u[..., 1:-1, 1:-1] + u[..., 1:-1, 2:])
        v_on_T = way * 0.5 * (v[..., 1:-1, 1:-1] + v[..., 2:, 1:-1])
        up = torch.where(u_on_T < 0, torch.tensor(0.0), u_on_T)
        um = torch.where(u_on_T > 0, torch.tensor(0.0), u_on_T)
        vp = torch.where(v_on_T < 0, torch.tensor(0.0), v_on_T)
        vm = torch.where(v_on_T > 0, torch.tensor(0.0), v_on_T)

        # PV advection
        rhs_q = self._adv(up, vp, um, vm, q0)
        rhs_q[..., 2:-2, 2:-2] -= way * (self.f[None, 3:-1, 2:-2] - self.f[None, 1:-3, 2:-2]) / (2 * self.dy) * 0.5 * (v[..., 2:-2, 2:-2] + v[..., 3:-1, 2:-2])
        
        # PV Diffusion
        if self.Kdiffus is not None:
            rhs_q[..., 2:-2, 2:-2] += (
                self.Kdiffus / (self.dx ** 2) * (q0[..., 2:-2, 3:-1] + q0[..., 2:-2, 1:-3] - 2 * q0[..., 2:-2, 2:-2]) +
                self.Kdiffus / (self.dy ** 2) * (q0[..., 3:-1, 2:-2] + q0[..., 1:-3, 2:-2] - 2 * q0[..., 2:-2, 2:-2])
            )
        rhs_q = torch.where(torch.isnan(rhs_q), torch.tensor(0.0), rhs_q)
        rhs_q[..., self.ind12] = 0
        rhs_q[..., self.ind0] = 0

        return rhs_q

    def _adv(self, up, vp, um, vm, var0):
        """
            3rd-order upwind scheme.
        """

        res = torch.zeros_like(var0, dtype=torch.float64)

        res[..., 2:-2,2:-2] = \
            - up[..., 1:-1, 1:-1] * 1 / (6 * self.dx) * \
            (2 * var0[..., 2:-2, 3:-1] + 3 * var0[..., 2:-2, 2:-2] - 6 * var0[..., 2:-2, 1:-3] + var0[..., 2:-2, :-4]) \
            + um[..., 1:-1, 1:-1] * 1 / (6 * self.dx) * \
            (var0[..., 2:-2, 4:] - 6 * var0[..., 2:-2, 3:-1] + 3 * var0[..., 2:-2, 2:-2] + 2 * var0[..., 2:-2, 1:-3]) \
            - vp[..., 1:-1, 1:-1] * 1 / (6 * self.dy) * \
            (2 * var0[..., 3:-1, 2:-2] + 3 * var0[..., 2:-2, 2:-2] - 6 * var0[..., 1:-3, 2:-2] + var0[..., :-4, 2:-2]) \
            + vm[..., 1:-1, 1:-1] * 1 / (6 * self.dy) * \
            (var0[..., 4:, 2:-2] - 6 * var0[..., 3:-1, 2:-2] + 3 * var0[..., 2:-2, 2:-2] + 2 * var0[..., 1:-3, 2:-2])

        return res

    def pv2h(self, q, hb, qb):
        """
        Potential Vorticity to SSH
        """
        qin = q[..., 1:-1, 1:-1] - qb[..., 1:-1, 1:-1]

        hrec = torch.zeros_like(q, dtype=torch.float64)
        inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
        hrec[..., 1:-1, 1:-1] = inv

        hrec += hb

        return hrec

    def step(self, h0, q0, hb, qb, way=1):

        # Compute geostrophic velocities
        u, v = self.h2uv(h0)
        
        # Compute increment
        incr = self.rhs(u,v,q0,way=way)
        
        # Time integration 
        q1 = q0 + way * self.dt * incr
        
        # Elliptical inversion 
        h1 = self.pv2h(q1, hb, qb)

        return h1, q1
    
    def forward(self, h0, hb, tint):
        """
        Forward model time integration

        Args:
            h0: Tensor of initial SSH field with shape (N, ny, nx)
            hb: Tensor of background SSH field with shape (N, ny, nx)
            tint: Time integration length

        Returns:
            h1: Tensor of final SSH field with shape (N, ny, nx)
        """
        q0 = self.h2pv(h0, hb)
        qb = self.h2pv(hb, hb)

        nstep = int(tint / self.dt)
        h1 = h0.clone()
        q1 = q0.clone()
        for _ in range(nstep):
            h1, q1 = self.step(h1, q1, hb, qb)
        
        return h1
