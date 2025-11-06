# uPIT SI-SNR(dB) for model optimization: two speakers.
class SISNR(object):
    def __init__(self):
        super(SISNR, self).__init__()

    def sisnr(self, estimate, target, eps=1e-8):
        estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
        target = target - tf.reduce_mean(target, axis=-1, keepdims=True)
        
        s_target = tf.reduce_sum(estimate * target, axis=-1, keepdims=True) * target / (tf.reduce_sum(target * target, axis=-1, keepdims=True) + eps)
        e_noise = estimate - s_target
        estimate_sisnr = 10 * tf.math.log(tf.reduce_sum(s_target * s_target, axis=-1, keepdims=True) + eps) / tf.math.log(10.0) - \
                    10 * tf.math.log(tf.reduce_sum(e_noise * e_noise, axis=-1, keepdims=True) + eps) / tf.math.log(10.0)

        return tf.squeeze(estimate_sisnr)


    def loss_calculate(self, estim, tar):
        tar = tf.transpose(tar, perm=[1, 0, 2])
        estim = tf.transpose(estim, perm=[1, 0, 2])
        def sisnr_loss(permute):
            return tf.reduce_mean(tf.stack([self.sisnr(estim[s], tar[t]) for s, t in enumerate(permute)]), axis = 0, keepdims = True)
        num_spks = estim.shape[0]
        sisnr_mat = tf.stack([sisnr_loss(p) for p in permutations(range(num_spks))])
        max_pmt = tf.reduce_max(sisnr_mat, axis=0)
        return -tf.reduce_mean(max_pmt)


# uPIT SI-SNR(dB) for model optimization: for a number of speakers.
def sisnr(estimated, target):
    tf.debugging.assert_equal(tf.shape(estimated), tf.shape(target), message="Estimated and target shapes must be equal.")
    B, C, T = estimated.shape
    mean_target = tf.reduce_mean(target, axis=2, keepdims=True)
    mean_estimate = tf.reduce_mean(estimated, axis=2, keepdims=True)
    zero_mean_target = target - mean_target
    zero_mean_estimate = estimated - mean_estimate
    s_target = tf.expand_dims(zero_mean_target, axis=1)  # [B, 1, C, T]
    s_estimate = tf.expand_dims(zero_mean_estimate, axis=2)  # [B, C, 1, T]
    pair_wise_dot = tf.reduce_sum(s_estimate * s_target, axis=3, keepdims=True)  # [B, C, C, 1]
    s_target_energy = tf.reduce_sum(tf.square(s_target), axis=3, keepdims=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    pair_wise_si_snr = tf.reduce_sum(tf.square(pair_wise_proj), axis=3) / (tf.reduce_sum(tf.square(e_noise), axis=3) + EPS)
    pair_wise_si_snr = 10 * tf.math.log(pair_wise_si_snr + EPS) / tf.math.log(10.0)
    perms_np = np.array(list(itertools.permutations(range(C))), dtype=np.int64)
    perms_tf = tf.constant(perms_np, dtype=tf.int32)
    perms_shape = tf.shape(perms_tf)
    num_perms = perms_shape[0]
    # one-hot, [C!, C, C]
    index = tf.expand_dims(perms_tf, axis=2)  # [C!, C, 1]
    perms_one_hot = tf.zeros((C, C, C), dtype=tf.float32)
    perms_one_hot = tf.tensor_scatter_nd_update(
        tf.zeros(tf.concat([perms_shape, [C]], axis=0), dtype=tf.float32),
        tf.stack([tf.tile(tf.expand_dims(tf.range(num_perms), axis=1), [1, C]),
                  tf.tile(tf.expand_dims(tf.range(C), axis=0), [num_perms, 1]),
                  perms_tf], axis=-1),
        tf.ones((num_perms, C), dtype=tf.float32)
    )
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = tf.einsum('bij,pij->bp', pair_wise_si_snr, perms_one_hot)
    max_pmt = tf.reduce_max(snr_set, axis=1, keepdims=True)
    max_pmt /= C
    return -tf.reduce_mean(max_pmt)