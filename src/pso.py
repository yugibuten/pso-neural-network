"""Luke, Essentials of Metaheuristics, Algorithm 39 (second edition), lines 1–28.
Source: https://people.cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf
Maximises fitness. Use negative loss to minimise. Informants are resampled per
particle per generation, without replacement, and always include self.
"""
import numpy as np


class PSO:
    def __init__(self, dim, fitness_fn, swarm_size=20, alpha=.7298, beta=1.496,
                 gamma=1.496, delta=0., epsilon=1., n_informants=3,
                 bounds=(-3., 3.), seed=0, boundary='clip', alpha_end=None):
        # Algorithm 39 lines 1–6: parameters; topology and bounds are explicit choices.
        if not isinstance(dim, int) or dim < 1 or not isinstance(swarm_size, int) or swarm_size < 1:
            raise ValueError('Positive integer dimension and swarm size required')
        if not isinstance(n_informants, int) or not 1 <= n_informants <= swarm_size:
            raise ValueError('n_informants includes self and must be in [1, swarm_size]')
        coefficients = [alpha, beta, gamma, delta, epsilon] + ([] if alpha_end is None else [alpha_end])
        if not np.isfinite(coefficients).all() or min(coefficients) < 0:
            raise ValueError('Coefficients must be finite and nonnegative')
        if boundary not in ('clip', 'reflect', 'reset'):
            raise ValueError('Unknown boundary handling')
        self.dim, self.fitness_fn, self.swarm_size = dim, fitness_fn, swarm_size
        self.alpha, self.beta, self.gamma, self.delta, self.epsilon = alpha, beta, gamma, delta, epsilon
        self.n_informants, self.seed, self.boundary, self.alpha_end = n_informants, seed, boundary, alpha_end
        self.low = np.broadcast_to(np.asarray(bounds[0], dtype=float), (dim,)).copy()
        self.high = np.broadcast_to(np.asarray(bounds[1], dtype=float), (dim,)).copy()
        if not np.isfinite([self.low, self.high]).all() or np.any(self.low >= self.high):
            raise ValueError('Finite increasing bounds required')

    def _bound(self, positions, velocities, rng):
        outside = (positions < self.low) | (positions > self.high)
        if self.boundary == 'clip':
            velocities[outside] = 0
            return np.clip(positions, self.low, self.high)
        if self.boundary == 'reset':
            positions = np.where(outside, rng.uniform(self.low, self.high, positions.shape), positions)
            velocities[outside] = 0
            return positions
        # Triangular-wave reflection handles multiple crossings in one step.
        width = self.high-self.low
        phase = np.mod(positions-self.low, 2*width)
        velocities *= np.where(phase > width, -1, 1)
        return self.low + np.where(phase <= width, phase, 2*width-phase)

    def run(self, iterations):
        """Return (best_vector, best_score, history).

        iterations counts assessed generations, including initialisation. Thus
        exactly swarm_size * iterations evaluations are made, with no unscored
        final movement. History contains one best-so-far score per generation.
        Repeated calls restart deterministically from seed.
        """
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError('iterations must be a positive integer')
        rng = np.random.default_rng(self.seed)
        # Lines 7–9: initialise swarm and random velocities.
        x = rng.uniform(self.low, self.high, (self.swarm_size, self.dim))
        v = rng.uniform(-.1*(self.high-self.low), .1*(self.high-self.low), x.shape)
        personal = x.copy()
        personal_scores = np.full(self.swarm_size, -np.inf)
        # Line 10: no best has yet been assessed.
        best, best_score, history = None, -np.inf, []
        self.n_evaluations = 0
        # Lines 11–13: assess all particles before calculating any movements.
        for generation in range(iterations):
            scores = np.array([float(self.fitness_fn(row.copy())) for row in x])
            self.n_evaluations += self.swarm_size
            if not np.isfinite(scores).all():
                raise ValueError('Fitness returned nonfinite score')
            improved = scores > personal_scores
            personal[improved], personal_scores[improved] = x[improved], scores[improved]
            # Lines 14–15: maximise, retaining a COPY of the historical best.
            index = int(np.argmax(personal_scores))
            if personal_scores[index] > best_score:
                best, best_score = personal[index].copy(), float(personal_scores[index])
            history.append(best_score)
            # Lines 27–28: budget termination; do not move to unassessed positions.
            if generation == iterations-1:
                break
            # Lines 16–19: personal, informant (including self), and global best.
            informant_best = np.empty_like(x)
            for j in range(self.swarm_size):
                others = np.delete(np.arange(self.swarm_size), j)
                informants = np.r_[j, rng.choice(others, self.n_informants-1, replace=False)]
                winner = informants[np.argmax(personal_scores[informants])]
                informant_best[j] = personal[winner]
            inertia = self.alpha
            if self.alpha_end is not None:
                inertia += (self.alpha_end-self.alpha)*generation/max(1, iterations-2)
            # Lines 20–24: independent random coefficients for every dimension.
            b = rng.uniform(0, self.beta, x.shape)
            c = rng.uniform(0, self.gamma, x.shape)
            d = rng.uniform(0, self.delta, x.shape)
            v = inertia*v + b*(personal-x) + c*(informant_best-x) + d*(best-x)
            # Lines 25–26: synchronous movement; boundary repair is an extension.
            x = self._bound(x+self.epsilon*v, v, rng)
        self.positions, self.velocities = x.copy(), v.copy()
        return best, best_score, np.asarray(history)


if __name__ == '__main__':
    solver = PSO(3, lambda x: -float(x @ x), swarm_size=30, seed=0, bounds=(-5, 5))
    position, score, curve = solver.run(200)
    print('Sphere loss:', -score)
    assert -score < 1e-5
    print('Sphere benchmark passed')
