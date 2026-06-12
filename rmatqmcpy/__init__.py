__version__ = "0.1"

from .tf_On_Un_SPn import (
    tf_On_QR,
    tf_Un_QR,
    tf_On_eig,
    tf_Un_eig,
    tf_Spn_SVD,
)

from .rand_On_Un_Spn import (
    rand_On_QR,
    rand_Un_QR,
    rand_On_eig,
    rand_Un_eig,
    rand_Spn_SVD,
)

from .rand_Stiefel_Gr_flag_LGr import (
    rand_flag_R,
    rand_flag_C,
    rand_flag_H,
    rand_LGr_R,
    rand_LGr_C,
    rand_LGr_H,
    rand_Stiefel_R,
    rand_Stiefel_C,
    rand_Stiefel_H,
    rand_Gr_R,
    rand_Gr_C,
    rand_Gr_H,
)

from .kernels import (
    KernelMatern,    
    KernelPoly,
)

from .opt import (
    opt_weights_sum_1,
    wce_squared_plus_kernel_integral,
)