import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, List, Tuple


#####################################
#          Boiler plate FEM         #
#####################################

def gauss_pt_eval(tensor: Tensor, N: List[Tensor], nsd: int = 2, stride: int = 1) -> Tensor:
    if nsd == 1:
        conv_gp = F.conv1d
    elif nsd == 2:
        conv_gp = F.conv2d
    elif nsd == 3:
        conv_gp = F.conv3d
    else:
        raise ValueError("nsd must be 1, 2, or 3")

    result_list = []
    for kernel in N:
        result = conv_gp(tensor, kernel, stride=stride)
        result_list.append(result)
    return torch.cat(result_list, dim=1)


class FEMEngine(nn.Module):
    """
    PDE Base Class
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.nsd = kwargs.get('nsd', 2)

        # For backward compatibility
        self.domain_length = kwargs.get('domain_length', 1.0)
        self.domain_size = kwargs.get('domain_size', 64)
        self.domain_lengths_nd = kwargs.get(
            'domain_lengths',
            (self.domain_length, self.domain_length, self.domain_length)
        )
        self.domain_sizes_nd = kwargs.get(
            'domain_sizes',
            (self.domain_size, self.domain_size, self.domain_size)
        )

        if self.nsd >= 2:
            self.domain_lengthX = self.domain_lengths_nd[0]
            self.domain_lengthY = self.domain_lengths_nd[1]
            self.domain_sizeX = self.domain_sizes_nd[0]
            self.domain_sizeY = self.domain_sizes_nd[1]
            if self.nsd >= 3:
                self.domain_lengthZ = self.domain_lengths_nd[2]
                self.domain_sizeZ = self.domain_sizes_nd[2]
                
        self.ngp_1d = kwargs.get('ngp_1d', 2)
        self.fem_basis_deg = kwargs.get('fem_basis_deg', 1)

        # Gauss quadrature setup
        if self.fem_basis_deg == 1:
            ngp_1d = 2
        elif self.fem_basis_deg in (2, 3):
            ngp_1d = 3
        else:
            raise ValueError("Unsupported fem_basis_deg. Supported degrees: 1, 2, 3.")

        if ngp_1d > self.ngp_1d:
            self.ngp_1d = ngp_1d

        self.ngp_total = self.ngp_1d ** self.nsd
        self.gpx_1d, self.gpw_1d = self.gauss_guadrature_scheme(self.ngp_1d)

        # Element setup
        self.nelemX = (self.domain_sizeX - 1) // self.fem_basis_deg
        self.nelemY = (self.domain_sizeY - 1) // self.fem_basis_deg
        if self.nsd == 3:
            self.nelemZ = (self.domain_sizeZ - 1) // self.fem_basis_deg
        self.nelem = (self.domain_size - 1) // self.fem_basis_deg  # Backward compatibility

        self.hx = self.domain_lengthX / self.nelemX
        self.hy = self.domain_lengthY / self.nelemY
        if self.nsd == 3:
            self.hz = self.domain_lengthZ / self.nelemZ
        self.h = self.domain_length / self.nelem  # Backward compatibility

        # Basis functions setup
        if self.fem_basis_deg == 1:
            # Linear basis functions
            self.nbf_1d = 2
            self.nbf_total = self.nbf_1d ** self.nsd

            self.bf_1d = lambda x: np.array([
                0.5 * (1.0 - x),
                0.5 * (1.0 + x)
            ])
            self.bf_1d_der = lambda x: np.array([
                -0.5,
                0.5
            ])
            self.bf_1d_der2 = lambda x: np.array([
                0.0,
                0.0
            ])

            self.bf_1d_th = lambda x: torch.stack([
                0.5 * (1.0 - x),
                0.5 * (1.0 + x)
            ])
            self.bf_1d_der_th = lambda x: torch.stack([
                -0.5 * torch.ones_like(x),
                0.5 * torch.ones_like(x)
            ])
            self.bf_1d_der2_th = lambda x: torch.stack([
                torch.zeros_like(x),
                torch.zeros_like(x)
            ])

        elif self.fem_basis_deg == 2:
            # Quadratic basis functions
            assert (self.domain_size - 1) % 2 == 0, \
                "For quadratic basis, (domain_size - 1) must be divisible by 2."
            self.nbf_1d = 3
            self.nbf_total = self.nbf_1d ** self.nsd

            self.bf_1d = lambda x: np.array([
                0.5 * x * (x - 1.0),
                1.0 - x ** 2,
                0.5 * x * (x + 1.0)
            ], dtype=float)
            self.bf_1d_der = lambda x: np.array([
                0.5 * (2.0 * x - 1.0),
                -2.0 * x,
                0.5 * (2.0 * x + 1.0)
            ], dtype=float)
            self.bf_1d_der2 = lambda x: np.array([
                1.0,
                -2.0,
                1.0
            ], dtype=float)

            self.bf_1d_th = lambda x: torch.stack([
                0.5 * x * (x - 1.0),
                1.0 - x ** 2,
                0.5 * x * (x + 1.0)
            ])
            self.bf_1d_der_th = lambda x: torch.stack([
                0.5 * (2.0 * x - 1.0),
                -2.0 * x,
                0.5 * (2.0 * x + 1.0)
            ])
            self.bf_1d_der2_th = lambda x: torch.stack([
                torch.ones_like(x),
                -2.0 * torch.ones_like(x),
                torch.ones_like(x)
            ])

        elif self.fem_basis_deg == 3:
            # Cubic basis functions
            assert (self.domain_size - 1) % 3 == 0, \
                "For cubic basis, (domain_size - 1) must be divisible by 3."
            self.nbf_1d = 4
            self.nbf_total = self.nbf_1d ** self.nsd

            self.bf_1d = lambda x: np.array([
                (-9.0 / 16.0) * (x ** 3 - x ** 2 - (1.0 / 9.0) * x + (1.0 / 9.0)),
                (27.0 / 16.0) * (x ** 3 - (1.0 / 3.0) * x ** 2 - x + (1.0 / 3.0)),
                (-27.0 / 16.0) * (x ** 3 + (1.0 / 3.0) * x ** 2 - x - (1.0 / 3.0)),
                (9.0 / 16.0) * (x ** 3 + x ** 2 - (1.0 / 9.0) * x - (1.0 / 9.0))
            ], dtype=float)
            self.bf_1d_der = lambda x: np.array([
                (-9.0 / 16.0) * (3.0 * x ** 2 - 2.0 * x - (1.0 / 9.0)),
                (27.0 / 16.0) * (3.0 * x ** 2 - (2.0 / 3.0) * x - 1.0),
                (-27.0 / 16.0) * (3.0 * x ** 2 + (2.0 / 3.0) * x - 1.0),
                (9.0 / 16.0) * (3.0 * x ** 2 + 2.0 * x - (1.0 / 9.0))
            ], dtype=float)

            self.bf_1d_der2 = lambda x: np.array([
                (-9.0 / 16.0) * (6.0 * x - 2.0),
                (27.0 / 16.0) * (6.0 * x - (2.0 / 3.0)),
                (-27.0 / 16.0) * (6.0 * x + (2.0 / 3.0)),
                (9.0 / 16.0) * (6.0 * x + 2.0)
            ], dtype=float)
                
                
    def gauss_guadrature_scheme(self, ngp_1d: int) -> Tuple[np.ndarray, np.ndarray]:
        if ngp_1d == 1:
            gpx_1d = np.array([0.0])
            gpw_1d = np.array([2.0])
        elif ngp_1d == 2:
            gpx_1d = np.array([-0.5773502691896258, 0.5773502691896258])
            gpw_1d = np.array([1.0, 1.0])
        elif ngp_1d == 3:
            gpx_1d = np.array([-0.774596669, 0.0, 0.774596669])
            gpw_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        elif ngp_1d == 4:
            gpx_1d = np.array([-0.861136, -0.339981, 0.339981, 0.861136])
            gpw_1d = np.array([0.347855, 0.652145, 0.652145, 0.347855])
        else:
            raise ValueError("Unsupported number of Gauss points per dimension.")
        return gpx_1d, gpw_1d

    def gauss_pt_evaluation(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(tensor, self.N_gp, nsd=self.nsd, stride=(self.nbf_1d - 1))

    def gauss_pt_evaluation_surf(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.N_gp_surf, nsd=(self.nsd - 1), stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der_x(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(tensor, self.dN_x_gp, nsd=self.nsd, stride=(self.nbf_1d - 1))

    def gauss_pt_evaluation_der_y(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(tensor, self.dN_y_gp, nsd=self.nsd, stride=(self.nbf_1d - 1))

    def gauss_pt_evaluation_der_z(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(tensor, self.dN_z_gp, nsd=self.nsd, stride=(self.nbf_1d - 1))

    def gauss_pt_evaluation_der2_x(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_x_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_y(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_y_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_z(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_z_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_xy(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_xy_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_yz(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_yz_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )

    def gauss_pt_evaluation_der2_zx(self, tensor: Tensor, stride: int = 1) -> Tensor:
        return gauss_pt_eval(
            tensor, self.d2N_zx_gp, nsd=self.nsd, stride=(self.nbf_1d - 1)
        )       
                
                

    def forward(self, *args: Tensor, **kwargs: Any) -> Tuple[Tensor, ...]:
        """
        Forward pass for FEMEngine.

        This method accepts an arbitrary number of input tensors and returns an arbitrary
        number of output tensors. It serves as a placeholder and should be implemented
        with the specific FEM computations during inference.

        Args:
            *args (Tensor): Variable length input tensor list representing the state or parameters.
            **kwargs (Any): Arbitrary keyworded input kwargs.

        Returns:
            Tuple[Tensor, ...]: A tuple of output tensors after FEM computations.
        """
        pass
    
    


class FEM2D(FEMEngine):
    """2D Finite Element Method Engine with Differentiable Components"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        assert self.nsd == 2, "FEM2D is designed for 2D problems only."

        self.gpw = torch.zeros(self.ngp_total)
        self.N_gp = nn.ParameterList()
        self.dN_x_gp = nn.ParameterList()
        self.dN_y_gp = nn.ParameterList()
        self.d2N_x_gp = nn.ParameterList()
        self.d2N_y_gp = nn.ParameterList()
        self.d2N_xy_gp = nn.ParameterList()

        # Initialize tensors to store basis functions and their derivatives
        self.Nvalues = torch.ones(
            (1, self.nbf_total, self.ngp_total, 1, 1)
        )
        self.dN_x_values = torch.ones(
            (1, self.nbf_total, self.ngp_total, 1, 1)
        )
        self.dN_y_values = torch.ones(
            (1, self.nbf_total, self.ngp_total, 1, 1)
        )
        self.d2N_x_values = torch.ones(
            (1, self.nbf_total, self.ngp_total, 1, 1)
        )
        self.d2N_y_values = torch.ones(
            (1, self.nbf_total, self.ngp_total, 1, 1)
        )
        self.d2N_xy_values = torch.ones(
            (1, self.nbf_total, self.ngp_total, 1, 1)
        )

        for jgp in range(self.ngp_1d):
            for igp in range(self.ngp_1d):
                N_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                dN_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                dN_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_x_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_y_gp = torch.zeros((self.nbf_1d, self.nbf_1d))
                d2N_xy_gp = torch.zeros((self.nbf_1d, self.nbf_1d))

                IGP = self.ngp_1d * jgp + igp  # Linear index for the Gauss point
                self.gpw[IGP] = self.gpw_1d[igp] * self.gpw_1d[jgp]

                for jbf in range(self.nbf_1d):
                    for ibf in range(self.nbf_1d):
                        IBF = self.nbf_1d * jbf + ibf
                        # Compute basis functions and their derivatives at Gauss points
                        bf_x = self.bf_1d(self.gpx_1d[igp])[ibf]
                        bf_y = self.bf_1d(self.gpx_1d[jgp])[jbf]
                        dbf_x = self.bf_1d_der(self.gpx_1d[igp])[ibf] * (2.0 / self.hx)
                        dbf_y = self.bf_1d_der(self.gpx_1d[jgp])[jbf] * (2.0 / self.hy)
                        d2bf_x = self.bf_1d_der2(self.gpx_1d[igp])[ibf] * (2.0 / self.hx) ** 2
                        d2bf_y = self.bf_1d_der2(self.gpx_1d[jgp])[jgp] * (2.0 / self.hy) ** 2
                        d2bf_xy = (
                            self.bf_1d_der(self.gpx_1d[igp])[ibf]
                            * self.bf_1d_der(self.gpx_1d[jgp])[jgp]
                            * (2.0 / self.hx)
                            * (2.0 / self.hy)
                        )

                        N_gp[jbf, ibf] = bf_x * bf_y
                        dN_x_gp[jbf, ibf] = dbf_x * bf_y
                        dN_y_gp[jbf, ibf] = bf_x * dbf_y
                        d2N_x_gp[jbf, ibf] = d2bf_x * bf_y
                        d2N_y_gp[jbf, ibf] = bf_x * d2bf_y
                        d2N_xy_gp[jbf, ibf] = dbf_x * dbf_y

                        # Store computed values
                        self.Nvalues[0, IBF, IGP, :, :] = N_gp[jbf, ibf]
                        self.dN_x_values[0, IBF, IGP, :, :] = dN_x_gp[jbf, ibf]
                        self.dN_y_values[0, IBF, IGP, :, :] = dN_y_gp[jbf, ibf]
                        self.d2N_x_values[0, IBF, IGP, :, :] = d2N_x_gp[jbf, ibf]
                        self.d2N_y_values[0, IBF, IGP, :, :] = d2N_y_gp[jbf, ibf]
                        self.d2N_xy_values[0, IBF, IGP, :, :] = d2N_xy_gp[jbf, ibf]

                # Append the computed kernels to the ParameterLists
                self.N_gp.append(
                    nn.Parameter(N_gp.unsqueeze(0).unsqueeze(1), requires_grad=False)
                )
                self.dN_x_gp.append(
                    nn.Parameter(dN_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False)
                )
                self.dN_y_gp.append(
                    nn.Parameter(dN_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False)
                )
                self.d2N_x_gp.append(
                    nn.Parameter(d2N_x_gp.unsqueeze(0).unsqueeze(1), requires_grad=False)
                )
                self.d2N_y_gp.append(
                    nn.Parameter(d2N_y_gp.unsqueeze(0).unsqueeze(1), requires_grad=False)
                )
                self.d2N_xy_gp.append(
                    nn.Parameter(d2N_xy_gp.unsqueeze(0).unsqueeze(1), requires_grad=False)
                )

        # Create spatial grid
        x = np.linspace(0, self.domain_lengthX, self.domain_sizeX)
        y = np.linspace(0, self.domain_lengthY, self.domain_sizeY)
        xx, yy = np.meshgrid(x, y)
        self.xx = torch.FloatTensor(xx)
        self.yy = torch.FloatTensor(yy)

        # Evaluate basis functions at grid points
        self.xgp = self.gauss_pt_evaluation(self.xx.unsqueeze(0).unsqueeze(0))
        self.ygp = self.gauss_pt_evaluation(self.yy.unsqueeze(0).unsqueeze(0))

        # Initialize local coordinates tensors
        self.xiigp = torch.ones_like(self.xgp)
        self.etagp = torch.ones_like(self.ygp)

        # Assign local coordinates to Gauss points
        for jgp in range(self.ngp_1d):
            for igp in range(self.ngp_1d):
                IGP = self.ngp_1d * jgp + igp
                self.xiigp[0, IGP, :, :] = torch.ones_like(
                    self.xiigp[0, IGP, :, :]
                ) * self.gpx_1d[igp]
                self.etagp[0, IGP, :, :] = torch.ones_like(
                    self.etagp[0, IGP, :, :]
                ) * self.gpx_1d[jgp]





#####################################
#          LDC NS Residual          #
#####################################


class ResidualLoss(FEM2D):
    '''
    Class for computing the residual of LDC-NS.
    
    Make sure to add device as an arg AND send to device as well (temporary hack)
    
    Example usage:
    
    # init data or anything else...
    
    # init model and residual
    model = model(*args)
    model.to(device)
    residual_fn = ResidualLoss(domain_size, device).to(device)
    
    # usage during inference
    for batch in test_dataloader:
        X, Y = batch
        Y_hat = model(X)
        residual_Total, div_Total = residual_fn(X, Y_hat)
    
    
    '''
    def __init__(self, domain_size, device, **kwargs):
        super().__init__(**kwargs)
        self.domain_size = domain_size
        self.hx = 2 / self.domain_size
        self.hy = 2 / self.domain_size
        self.device = device
        
    def forward(self, dataX, dataY):
        '''
        args:
            dataX: SDF or Mask and Re
            dataY: Field solution (u, v, p)
        '''
        r = dataX[:,0:1,...]        
        s = dataX[:,1:2,...]  # SDF (Signed Distance Function)
        u = dataY[:,0:1,...]
        v = dataY[:,1:2,...]
        p = dataY[:,2:3,...]

        # Apply mask where SDF < 0 (inside the geometry)
        mask = (s > 0).float()  # Assuming s is SDF, mask outside geometry

        # Apply mask to predicted fields
        u = u * mask
        v = v * mask
        p = p * mask
        
        gpw = self.gpw
        trnsfrm_jac = (0.5*self.hx)*(0.5*self.hy)
        JxW = (gpw*trnsfrm_jac).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(self.device)

        u_gp = self.gauss_pt_evaluation(u)
        v_gp = self.gauss_pt_evaluation(v)
        u_x_gp = self.gauss_pt_evaluation_der_x(u)
        u_y_gp = self.gauss_pt_evaluation_der_y(u)
        v_x_gp = self.gauss_pt_evaluation_der_x(v)
        v_y_gp = self.gauss_pt_evaluation_der_y(v)
        p_x_gp = self.gauss_pt_evaluation_der_x(p)
        p_y_gp = self.gauss_pt_evaluation_der_y(p)

        r1_gp = (u_gp * u_x_gp + v_gp * u_y_gp + p_x_gp)**2
        r2_gp = (u_gp * v_x_gp + v_gp * v_y_gp + p_y_gp)**2

        r1_elm_gp = r1_gp * JxW
        r2_elm_gp = r2_gp * JxW

        r1_elm = torch.sum(r1_elm_gp, 1)
        r2_elm = torch.sum(r2_elm_gp, 1)
        r1_total_squared = torch.sum(torch.sum(r1_elm, -1), -1)
        r2_total_squared = torch.sum(torch.sum(r2_elm, -1), -1)
        #r_Total = (r1_total_squared + r2_total_squared).pow(2).mean() / (self.domain_size**2)  # Should perhaps be pow(0.5) to take th square root 
        r_Total = (r1_total_squared + r2_total_squared).pow(2).mean() / (self.domain_size**2)
        div_gp = u_x_gp + v_y_gp
        div_elm_gp = div_gp * JxW
        div_elm = torch.sum(div_elm_gp, 1)
        div_Total = torch.sum(torch.sum(div_elm, -1), -1)
        
        return r1_elm, r2_elm, div_elm, r_Total, div_Total