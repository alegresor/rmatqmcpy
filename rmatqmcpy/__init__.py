__version__ = "0.1"

from .tf_coe_cue_cqe import (
    tf_coe_qr,
    tf_cue_qr,
    tf_coe_eig,
    tf_cue_eig,
    tf_cqe_svd,
)

from .rand_coe_cue_cqe import (
    rand_coe_qr,
    rand_cue_qr,
    rand_coe_eig,
    rand_cue_eig,
    rand_cqe_svd,
)

from .rand_stiefel_gr_flag_lgr import (
    rand_flag_real,
    rand_flag_complex,
    rand_flag_quaternionic,
    rand_lgr_real,
    rand_lgr_complex,
    rand_lgr_quaternionic,
    rand_stiefel_real,
    rand_stiefel_complex,
    rand_stiefel_quaternionic,
    rand_gr_real,
    rand_gr_complex,
    rand_gr_quaternionic,
)

from .kernels import (
    KernelMatern,    
    KernelPoly,
)

from .opt import (
    opt_weights_sum_1,
)