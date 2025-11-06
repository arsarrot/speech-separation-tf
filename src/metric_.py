# uPIT SI-SNRi(dB): two speakers.
class SISNRi_tf(object):
    def __init__(self):
        super(SISNRi_tf, self).__init__()

    def sisnri(self, mixture, estimate, target, eps=1e-8):
        estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
        target = target - tf.reduce_mean(target, axis=-1, keepdims=True)
        mixture = mixture - tf.reduce_mean(mixture, axis=-1, keepdims=True)

        s_target = tf.reduce_sum(estimate * target, axis=-1, keepdims=True) * target / (tf.reduce_sum(target * target, axis=-1, keepdims=True) + eps)
        e_noise = estimate - s_target
        mixture_p = tf.reduce_sum(mixture * target, axis=-1, keepdims=True) * target / (tf.reduce_sum(target * target, axis=-1, keepdims=True) + eps)
        mixture_v = mixture - mixture_p

        est_sisnr = 10 * tf.math.log(tf.reduce_sum(s_target * s_target, axis=-1, keepdims=True) + eps) / tf.math.log(10.0) - \
                    10 * tf.math.log(tf.reduce_sum(e_noise * e_noise, axis=-1, keepdims=True) + eps) / tf.math.log(10.0)

        mix_sisnr = 10 * tf.math.log(tf.reduce_sum(mixture_p * mixture_p, axis=-1, keepdims=True) + eps) / tf.math.log(10.0) - \
                    10 * tf.math.log(tf.reduce_sum(mixture_v * mixture_v, axis=-1, keepdims=True) + eps) / tf.math.log(10.0)

        return tf.squeeze(est_sisnr - mix_sisnr)


    def calculate_score(self, mixtures, estim, tar):
        tar = tf.transpose(tar, perm=[1, 0, 2])
        estim = tf.transpose(estim, perm=[1, 0, 2])
        def sisnr_loss(permute):
            return tf.reduce_mean(tf.stack([self.sisnri(mixtures, estim[s], tar[t]) for s, t in enumerate(permute)]), axis = 0, keepdims = True)
        num_spks = estim.shape[0]
        sisnr_mat = tf.stack([sisnr_loss(p) for p in permutations(range(num_spks))])
        max_pmt = tf.reduce_max(sisnr_mat, axis=0)
        return tf.reduce_mean(max_pmt)



# uPIT SI-SNRi(dB) for estimated speaker 0.
class SISNRi_tf(object):
    def __init__(self):
        super(SISNRi_tf, self).__init__()

    def sisnri(self, mixture, estimate, target, eps=1e-8):
        estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
        target = target - tf.reduce_mean(target, axis=-1, keepdims=True)
        mixture = mixture - tf.reduce_mean(mixture, axis=-1, keepdims=True)

        s_target = tf.reduce_sum(estimate * target, axis=-1, keepdims=True) * target / (tf.reduce_sum(target * target, axis=-1, keepdims=True) + eps)
        e_noise = estimate - s_target
        mixture_p = tf.reduce_sum(mixture * target, axis=-1, keepdims=True) * target / (tf.reduce_sum(target * target, axis=-1, keepdims=True) + eps)
        mixture_v = mixture - mixture_p

        est_sisnr = 10 * tf.math.log(tf.reduce_sum(s_target * s_target, axis=-1, keepdims=True) + eps) / tf.math.log(10.0) - \
                    10 * tf.math.log(tf.reduce_sum(e_noise * e_noise, axis=-1, keepdims=True) + eps) / tf.math.log(10.0)

        mix_sisnr = 10 * tf.math.log(tf.reduce_sum(mixture_p * mixture_p, axis=-1, keepdims=True) + eps) / tf.math.log(10.0) - \
                    10 * tf.math.log(tf.reduce_sum(mixture_v * mixture_v, axis=-1, keepdims=True) + eps) / tf.math.log(10.0)

        return tf.squeeze(est_sisnr - mix_sisnr)

    def calculate_score(self, mixtures, estim, tar):
        tar = tf.transpose(tar, perm=[1, 0, 2])
        estim = tf.transpose(estim, perm=[1, 0, 2])
        def sisnr_loss(permute):
            return [self.sisnri(mixtures, estim[t], tar[s]) for s, t in enumerate(permute)]
        sisnr_mat = ([sisnr_loss((0, 0))]) # (0,0) is the estimated source index. That means, the for loop in sisnr_loss would be iterated twice with (estim0, tar0) and (estim0, tar1)
        max_pmt = tf.reduce_max(sisnr_mat)
        return max_pmt


# uPIT SI-SNRi(dB) for estimated speaker 1.
class SISNRi_tf(object):
    def __init__(self):
        super(SISNRi_tf, self).__init__()

    def sisnri(self, mixture, estimate, target, eps=1e-8):
        estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
        target = target - tf.reduce_mean(target, axis=-1, keepdims=True)
        mixture = mixture - tf.reduce_mean(mixture, axis=-1, keepdims=True)
        
        s_target = tf.reduce_sum(estimate * target, axis=-1, keepdims=True) * target / (tf.reduce_sum(target * target, axis=-1, keepdims=True) + eps)
        e_noise = estimate - s_target
        mixture_p = tf.reduce_sum(mixture * target, axis=-1, keepdims=True) * target / (tf.reduce_sum(target * target, axis=-1, keepdims=True) + eps)
        mixture_v = mixture - mixture_p

        est_sisnr = 10 * tf.math.log(tf.reduce_sum(s_target * s_target, axis=-1, keepdims=True) + eps) / tf.math.log(10.0) - \
                    10 * tf.math.log(tf.reduce_sum(e_noise * e_noise, axis=-1, keepdims=True) + eps) / tf.math.log(10.0)

        mix_sisnr = 10 * tf.math.log(tf.reduce_sum(mixture_p * mixture_p, axis=-1, keepdims=True) + eps) / tf.math.log(10.0) - \
                    10 * tf.math.log(tf.reduce_sum(mixture_v * mixture_v, axis=-1, keepdims=True) + eps) / tf.math.log(10.0)

        return tf.squeeze(est_sisnr - mix_sisnr)

    def calculate_score(self, mixtures, estim, tar):
        tar = tf.transpose(tar, perm=[1, 0, 2])
        estim = tf.transpose(estim, perm=[1, 0, 2])
        def sisnr_loss(permute):
            return [self.sisnri(mixtures, estim[t], tar[s]) for s, t in enumerate(permute)]
        sisnr_mat = ([sisnr_loss((1, 1))]) # (1,1) is the estimated source index. That means, the for loop in sisnr_loss would be iterated twice with (estim1, tar0) and (estim1, tar1)
        max_pmt = tf.reduce_max(sisnr_mat)
        return max_pmt