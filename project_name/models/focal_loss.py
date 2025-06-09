import typing

import tensorflow as tf
from keras.losses import Loss


class MyBinaryFocalCrossentropy(Loss):
    """Implements binary focal crossentropy per class. Suitable for multi-label classification."""

    def __init__(self, alpha: tf.Tensor, gamma: tf.Tensor) -> None:
        """Create a binary focal crossentropy loss function.

        Args:
            alpha (tf.Tensor): controls weighting. Must be in [0,1].
            gamma (tf.Tensor): controls focusing. Must be nonnegative.
        """
        super().__init__()
        self.alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)          # input validation omitted
        self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)          # input validation omitted

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute focal crossentropy loss per label and aggregate."""
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)                 # y_pred if y_true, 1 - y_pred otherwise
        focus = tf.math.pow(1 - p_t, self.gamma)                            # 0^0 := 1, no need to correct

        eps = tf.keras.backend.epsilon()
        p_t = tf.clip_by_value(p_t, eps, 1 - eps)                           # numerical stability with log
        ce = -tf.math.log(p_t)

        alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)     # alpha if y_true, 1 - alpha otherwise
        weighted_fl = alpha_t * focus * ce

        return tf.reduce_mean(weighted_fl, axis=-1)                         # aggregate over labels