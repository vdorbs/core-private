from keras import Model
from keras.layers import Add, Concatenate, Dot, Dense, Input, Reshape

from .keras_residual_affine_model import KerasResidualAffineModel

class KerasResidualRoboticModel(KerasResidualAffineModel):
    def __init__(self, n, d_hidden, m):
        qs = Input((n,))
        shared_reps = Dense(d_hidden, activation='relu')(qs)
        shared_model = Model(qs, shared_reps)

        act_reps = Dense(d_hidden, activation='relu')(shared_reps)
        act_output_vecs = Dense(n * m)(act_reps)
        act_outputs = Reshape((n, n))(act_output_vecs)
        act_model = Model(qs, act_outputs)

        xs = Input((2 * n,))
        # TODO: Fix below
        shared_reps = shared_model(xs[:n])
        # TODO: Fix above
        drift_inputs = Concatenate()([xs, shared_reps])
        drift_reps = Dense(d_hidden, activation='relu')(drift_inputs)
        drift_outputs = Dense(n)(drift_reps)
        drift_model = Model(xs, drift_outputs)

        us = Input((m,))
        outputs = Add()([drift_model, Dot([2, 1])([act_model, us])])
        model = Model([qs, xs, us], outputs)

        KerasResidualAffineModel.__init__(self, drift_model, act_model, model)
