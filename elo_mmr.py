import math
import statistics

# params
# β2 is variance of performance, adjust based on correspondence btw given rating gap and win rate (or just to stretch)
# β2 = 200. ** 2  # β=200 default
β2 = 200 ** 2 / (2 * statistics.NormalDist().inv_cdf(2./3) ** 2)  # β=328 imo nicer, empirically is closer to cf/dmoj
# β2 = 400 ** 2 / (2 * statistics.NormalDist().inv_cdf(10./11) ** 2)  # β=212 cf/elo
# β2 = 200 ** 2 / (2 * statistics.NormalDist().inv_cdf(.75) ** 2)  # β=210 original elo
print(f'{β2**.5=}')
# exit(0)
# (μ_init, σ2_init) give initial (normal) prior on rating.
# μ_init just shifts results (though displayed rating may be adjusted for new players)
# high/low σ2_init can be used to adjust more quickly/slowly to fair ratings
# these seem to be reasonable priors based on cf rating distr
μ_init = 1500.  # default
σ2_init = 350. ** 2 * (β2 / 212**2)  # default
σ_init = σ2_init ** .5
# ρ is how quickly to move from logistic to gaussian belief
# lower values are thus supposed to be more robust to big one-time deviations, higher values allow remembering less
# mostly important to be positive, finite for theoretical reasons, ρ=1 fine (but doesn't matter much)
ρ = 1.
# uncertainty added passively between contests, due to off-site practice/oxidation
var_per_second = 0.
# uncertainty added between contests (maybe improving during the actual contest, or implied practice)
var_per_contest = 1219.047619 * (β2 / 212**2)

# everyone should quickly converge to this uncertainty level
# we derive from var_per_contest, Elo-MMR does the reverse
σ2_lim = ((var_per_contest**2 + 4*β2*var_per_contest) ** .5 - var_per_contest) / 2
# original values:
# σ2_lim = 80. ** 2
# var_per_contest = σ2_lim * σ2_lim / (β2 - σ2_lim)

VALID_RANGE = μ_init - 20*σ_init, μ_init + 20*σ_init
TANH_C = 3 ** .5 / math.pi

class HistoryEntry:
    def __init__(self, rating, rlo, rhi, μ, σ, p, ts, weight):
        self.rating = rating
        self.rlo = rlo
        self.rhi = rhi
        self.μ = μ
        self.σ = σ
        self.p = p
        self.ts = ts
        self.weight = weight
    def __repr__(self):
        return f'[{self.rating} {self.rlo} {self.rhi} {round(self.μ)} {round(self.σ)} {round(self.p)}]'

def eval_tanhs(tanh_terms, x):
    return sum((wt / (TANH_C * σ)) * math.tanh((x - μ) / (2 * TANH_C * σ)) for μ, σ, wt in tanh_terms)

def eval_tanhs_grad(tanh_terms, x):
    return sum(0.5 * wt * (TANH_C * σ * math.cosh((x - μ) / (2 * TANH_C * σ))) ** -2 for μ, σ, wt in tanh_terms)

# TODO smooth, can run Newton instead
        # let z = (x - self.mu) * self.w_arg;
        # let val = -z.tanh() * self.w_out;
        # let val_prime = -z.cosh().powi(-2) * self.w_arg * self.w_out;

def solve_newton(tanh_terms, y_tg=0, lin_factor=0):
    x = μ_init
    # print(tanh_terms, x, y_tg, lin_factor)
    it=0
    while True:
        it+=1
        y = lin_factor * x + eval_tanhs(tanh_terms, x)
        if abs(y-y_tg) < 1e-4:
            break
        if it>5:
            print(tanh_terms, y_tg, lin_factor, x, y)
            return solve_slo(tanh_terms, y_tg, lin_factor)
        grad = lin_factor + eval_tanhs_grad(tanh_terms, x)
        x += (y_tg - y) / grad
    # print(x, y, grad, solve_slo(tanh_terms, y_tg, lin_factor))
    return x

def solve(tanh_terms, y_tg=0, lin_factor=0, bounds=VALID_RANGE):
    L,R = bounds
    while R-L > 1e-5*σ_init:
        x = 0.5 * (L + R)
        y = lin_factor * x + eval_tanhs(tanh_terms, x)
        if y > y_tg:
            R = x
        elif y < y_tg:
            L = x
        else:
            return x
    return (L + R) / 2


# contest_time in seconds
# standings: [(user, rk_lo, rk_hi)], where rk_lo, rk_hi = rk, rk+t-1 when t people tie for rank rk
# user_histories: {user: [history_entry]}, for at least any users who have played before, only past histories
# only need O(1) most recent rateds, relevance drops off exponentially (and all should be fine/stable just dropping)
# can use weight=0 for unrated, weight=1 for rated.
# returns {user: [history_entry]}, with new histories added, modified in-place
def rate(contest_key, contest_time, standings, user_histories, weight=1.):
    n = len(standings)

    μ = [μ_init] * n
    δ = [0] * n
    for i, (u, rlo, rhi) in enumerate(standings):
        dt = 0
        hist = user_histories[u]
        σ2 = σ2_init
        if len(hist):
            μ[i] = hist[-1].μ
            σ2 = hist[-1].σ ** 2
            dt = contest_time - hist[-1].ts
        γ2 = weight * var_per_contest + dt * var_per_second
        σ2 += γ2
        σ2_perf = σ2_lim + (β2 - σ2_lim) / weight
        δ[i] = (σ2 + σ2_perf) ** .5
        new_σ = (1. / σ2 + 1. / σ2_perf) ** -.5
        hist.append(HistoryEntry(0, rlo, rhi, 0, new_σ, 0, contest_time, weight))

    if True:
        tanh_terms = list(zip(μ, δ, [1]*n))
        y_tg_pfx = [0.]*(n+1)
        for i, (_, rlo, _) in enumerate(standings):
            y_tg_pfx[rlo] += 1. / (TANH_C * δ[i])
        for i in range(n):
            y_tg_pfx[i+1] += y_tg_pfx[i]
        y_tgs = sorted(((y_tg_pfx[n] - y_tg_pfx[rhi] - y_tg_pfx[rlo-1], u) for u, rlo, rhi in standings), key=lambda p:p[0])
        def divconq(l, r, L, R):
            if l > r:
                return
            m = (l + r) // 2
            y_tg = y_tgs[m][0]
            M = solve(tanh_terms, y_tg, bounds=(L,R))
            while m > l and y_tgs[m-1][0] == y_tg:
                m -= 1
            divconq(l, m-1, L, M)
            while m <= r and y_tgs[m][0] == y_tg:
                user_histories[y_tgs[m][1]][-1].p = M
                m += 1
            divconq(m, r, M, R)
        divconq(0, n-1, solve(tanh_terms, y_tgs[0][0]), solve(tanh_terms, y_tgs[-1][0]))
        # divconq(0, n-1, *VALID_RANGE)
        # bounds = solve(tanh_terms, y_tgs[0][0]), solve(tanh_terms, y_tgs[-1][0])
        # for y_tg, u in y_tgs:
            # user_histories[u][-1].p = solve(tanh_terms, y_tg)
    else:
        for i, (u, rlo, rhi) in enumerate(standings):
            assert 1 <= rlo <= rhi <= n
            tanh_terms = []
            y_tg = 0
            for j, (_, orlo, orhi) in enumerate(standings):
                if orlo > rhi:  # j loses to i
                    tanh_terms.append((μ[j], δ[j], 1))
                    y_tg += 1. / (TANH_C * δ[j])
                elif orhi < rlo:  # j beats i
                    tanh_terms.append((μ[j], δ[j], 1))
                    y_tg -= 1. / (TANH_C * δ[j])
                else:  # tie
                    assert (rlo, rhi) == (orlo, orhi)
                    # This counts a tie as a win and a loss, as per Elo-MMR.
                    # TODO More typical choice (allowing reuse of this list) would be 0.5W+0.5L.
                    tanh_terms.append((μ[j], δ[j], 1))
                    # tanh_terms.append((μ[j], δ[j], 1))
            user_histories[u][-1].p = round(solve(tanh_terms, y_tg))
            # TODO if we do .5W+.5L, optimize above to divconq bsearch.  loop until R-L<σ_init/1e3 or smth each.

    for i, (u, rlo, rhi) in enumerate(standings):
        hist = user_histories[u]
        tanh_terms = []
        ω_prev = 1.
        t_prev = contest_time
        wt_prev = 0.
        w_sum = 0.
        for h in user_histories[u][::-1]:
            γ2 = wt_prev * var_per_contest + (t_prev - h.ts) * var_per_second
            t_prev = h.ts
            wt_prev = h.weight
            κ = h.σ**2 / (h.σ**2 + γ2)
            ω = ω_prev * κ**(1+ρ)
            if ω < 1e-3:  # TODO can prune some
                # default params, this is roughly equiv to taking only 20 recentest rateds each, result appears unch
                print(len(tanh_terms),len(user_histories[u]))
                break
            σ2_perf = σ2_lim + (β2 - σ2_lim) / h.weight
            tanh_terms.append((h.p, σ2_perf ** .5, ω))
            ω_prev = ω
            w_sum += ω / σ2_perf
        w0 = hist[-1].σ**-2 - w_sum
        assert w0 > 0
        p0 = eval_tanhs(tanh_terms[1:], μ[i]) / w0 + μ[i]
        hist[-1].μ = solve(tanh_terms, w0*p0, lin_factor=w0)
        # hist[-1].μ = round(hist[-1].μ)
        # hist[-1].σ = round(hist[-1].σ)
        hist[-1].rating = round(hist[-1].μ - 2*hist[-1].σ)
        hist[-1].rating = round(hist[-1].μ)
        hist[-1].rating = round(hist[-1].μ - 2*hist[-1].σ + 2*σ2_lim**.5)
