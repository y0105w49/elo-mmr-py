import math


μ_init = 1500.
σ2_init = 350. ** 2
β2 = 200. ** 2
σ2_lim = 80. ** 2
ρ = 1.
var_per_second = 0.
var_per_contest = σ2_lim * σ2_lim / (β2 - σ2_lim)
TANH_C = 3 ** .5 / math.pi

class HistoryEntry:
    def __init__(self, rating, rlo, rhi, μ, σ, p, β, ts, weight):
        self.rating = rating
        self.rlo = rlo
        self.rhi = rhi
        self.μ = μ
        self.σ = σ
        self.p = p
        self.β = β
        self.ts = ts
        self.weight = weight
    def __repr__(self):
        return f'{self.rating} {self.rlo} {self.rhi} {self.μ} {self.σ} {self.p} {self.β}'

def eval_tanhs(tanh_terms, x):
    return sum((wt / (TANH_C * σ)) * math.tanh((x - μ) / (2 * TANH_C * σ)) for μ, σ, wt in tanh_terms)

# TODO smooth, can run Newton instead
def solve(tanh_terms, y_tg=0, lin_factor=0):
    L, R = -6000, 9000
    for _ in range(20):
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
# can use weight=0 for unrated, weight=1 for rated.
# returns {user: [history_entry]}, with new histories added, modified in-place
def rate(contest_key, contest_time, standings, user_histories, weight=1.):
    σ2_perf = σ2_lim + (β2 - σ2_lim) / weight
    σ_perf = σ2_perf ** .5

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
        δ[i] = (σ2 + σ2_perf) ** .5
        new_σ = (1. / σ2 + 1. / σ2_perf) ** -.5
        hist.append(HistoryEntry(0, rlo, rhi, 0, new_σ, 0, σ_perf, contest_time, weight))

    for i, (u, rlo, rhi) in enumerate(standings):
        assert rlo <= rhi
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
                tanh_terms.append((μ[j], δ[j], 1))
        user_histories[u][-1].p = round(solve(tanh_terms, y_tg))

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
            if ω < 0:
                break
            tanh_terms.append((h.p, h.β, ω))
            ω_prev = ω
            w_sum += ω / h.β**2
        w0 = hist[-1].σ**-2 - w_sum
        assert w0 > 0
        p0 = eval_tanhs(tanh_terms[1:], μ[i]) / w0 + μ[i]
        hist[-1].μ = solve(tanh_terms, w0*p0, lin_factor=w0)
        # hist[-1].μ = round(hist[-1].μ)
        # hist[-1].σ = round(hist[-1].σ)
        hist[-1].rating = round(hist[-1].μ - 2*hist[-1].σ)
        hist[-1].rating = round(hist[-1].μ)
        hist[-1].rating = round(hist[-1].μ - 2*hist[-1].σ + 2*σ2_lim**.5)
