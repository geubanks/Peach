/* See engine.h for what this file is and why it is in C.
 *
 * Written from ../watch/CLAUDE.md section 3. Points where that text did not
 * determine the answer are marked SPEC GAP, with the choice made and the
 * reason. Those marks are the deliverable: each one is a place the Swift
 * author would also have had to guess, and a coin-flip that fails parity.
 */

#include "engine.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void tone_config_defaults(ToneConfig *cfg) {
    cfg->artifact_threshold = 0.20;
    cfg->min_intervals = 30;
    cfg->baseline_days = 28.0;
    cfg->min_baseline_windows = 30;
    cfg->min_baseline_hours = 5;
    cfg->min_scoring_windows = 30;
    cfg->lambda_hrv = 1.0;
    cfg->sigma2_eps_x = -1.0;
    cfg->sigma2_eps_h = -1.0;
    cfg->mad_scale = 1.4826;
    cfg->interval_is_t = 1;
}

/* ---------------------------------------------------------------- Step 1 */

void tone_filter_rr(const double *rr, int n, double threshold, unsigned char *keep) {
    if (n <= 0) return;
    if (n == 1) { keep[0] = 1; return; }
    for (int k = 0; k < n; k++) {
        int ok = 0;
        if (k > 0 && fabs(rr[k] - rr[k - 1]) / rr[k - 1] <= threshold) ok = 1;
        if (!ok && k + 1 < n && fabs(rr[k] - rr[k + 1]) / rr[k + 1] <= threshold) ok = 1;
        keep[k] = (unsigned char)ok;
    }
}

double tone_rmssd(const double *rr, int n, const unsigned char *keep, int *n_diffs) {
    double sum = 0.0;
    int d = 0;
    for (int k = 0; k + 1 < n; k++) {
        if (keep[k] && keep[k + 1]) {
            double diff = rr[k + 1] - rr[k];
            sum += diff * diff;
            d++;
        }
    }
    if (n_diffs) *n_diffs = d;
    if (d == 0) return NAN;
    return sqrt(sum / (double)d);
}

double tone_mean_rr(const double *rr, int n, const unsigned char *keep) {
    /* SPEC GAP 1. Step 2 says "60000 divided by the mean interval" but does not
     * say whether the mean is over all intervals or only the kept ones. Using
     * all of them would put the ectopic beat and its compensatory pause -- the
     * very intervals Step 1 just rejected -- back into the heart rate. Kept
     * only. (They nearly cancel, so the error is small and would have shown up
     * as a mysterious ~0.1% parity failure rather than an obvious one.) */
    double sum = 0.0;
    int count = 0;
    for (int k = 0; k < n; k++) {
        if (keep[k]) { sum += rr[k]; count++; }
    }
    if (count == 0) return NAN;
    return sum / (double)count;
}

ToneMetrics tone_window_metrics(const double *rr, int n, const ToneConfig *cfg) {
    ToneMetrics m;
    memset(&m, 0, sizeof(m));
    m.rmssd = m.mean_hr = NAN;
    if (n <= 0) return m;

    for (int k = 0; k < n; k++) {
        if (!isfinite(rr[k]) || rr[k] <= 0.0) return m;
    }

    unsigned char *keep = malloc((size_t)n);
    if (!keep) return m;
    tone_filter_rr(rr, n, cfg->artifact_threshold, keep);

    for (int k = 0; k < n; k++) m.n_kept += keep[k];
    m.rmssd = tone_rmssd(rr, n, keep, &m.n_diffs);
    double mean_rr = tone_mean_rr(rr, n, keep);
    m.mean_hr = (isfinite(mean_rr) && mean_rr > 0.0) ? 60000.0 / mean_rr : NAN;
    free(keep);

    m.usable = (m.n_kept >= cfg->min_intervals)
            && (m.n_diffs >= 2)
            && isfinite(m.rmssd) && m.rmssd > 0.0
            && isfinite(m.mean_hr) && m.mean_hr > 0.0;
    return m;
}

/* ---------------------------------------------------------------- Step 3 */

int tone_solve3(const double ata[9], const double atb[3], double out[3]) {
    double a[9], b[3];
    memcpy(a, ata, sizeof(a));
    memcpy(b, atb, sizeof(b));

    for (int col = 0; col < 3; col++) {
        int pivot = col;
        for (int row = col + 1; row < 3; row++)
            if (fabs(a[row * 3 + col]) > fabs(a[pivot * 3 + col])) pivot = row;
        if (fabs(a[pivot * 3 + col]) < TONE_PIVOT_EPS) return 0;
        if (pivot != col) {
            for (int j = 0; j < 3; j++) {
                double t = a[col * 3 + j]; a[col * 3 + j] = a[pivot * 3 + j]; a[pivot * 3 + j] = t;
            }
            double t = b[col]; b[col] = b[pivot]; b[pivot] = t;
        }
        for (int row = col + 1; row < 3; row++) {
            double factor = a[row * 3 + col] / a[col * 3 + col];
            for (int j = col; j < 3; j++) a[row * 3 + j] -= factor * a[col * 3 + j];
            b[row] -= factor * b[col];
        }
    }
    for (int row = 2; row >= 0; row--) {
        double s = b[row];
        for (int j = row + 1; j < 3; j++) s -= a[row * 3 + j] * out[j];
        out[row] = s / a[row * 3 + row];
    }
    return 1;
}

static void flat_fit(const double *y, int n, ToneFit *fit) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += y[i];
    fit->mesor = (n > 0) ? sum / (double)n : NAN;
    fit->a = fit->b = 0.0;
    fit->n = n;
    fit->flat = 1;
}

void tone_cosinor_fit(const double *hours, const double *y, int n,
                      int min_distinct_hours, ToneFit *fit) {
    /* SPEC GAP 2. "fewer than 5 distinct clock hours" does not say how an hour
     * is counted. Chosen: floor(hour) reduced mod 24, i.e. the integer hour
     * bucket. The alternative -- counting distinct fractional hours -- would
     * make the guard useless, since two samples are almost never at the same
     * second. */
    if (n < 4) { flat_fit(y, n, fit); return; }

    if (min_distinct_hours > 0) {
        unsigned char seen[24];
        memset(seen, 0, sizeof(seen));
        for (int i = 0; i < n; i++) {
            int bucket = (int)floor(hours[i]) % 24;
            if (bucket < 0) bucket += 24;
            seen[bucket] = 1;
        }
        int distinct = 0;
        for (int i = 0; i < 24; i++) distinct += seen[i];
        if (distinct < min_distinct_hours) { flat_fit(y, n, fit); return; }
    }

    double ata[9], atb[3], beta[3];
    memset(ata, 0, sizeof(ata));
    memset(atb, 0, sizeof(atb));
    for (int i = 0; i < n; i++) {
        double w = TONE_TWO_PI_OVER_24 * hours[i];
        double row[3] = { 1.0, cos(w), sin(w) };
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) ata[r * 3 + c] += row[r] * row[c];
            atb[r] += row[r] * y[i];
        }
    }
    if (!tone_solve3(ata, atb, beta)
        || !isfinite(beta[0]) || !isfinite(beta[1]) || !isfinite(beta[2])) {
        flat_fit(y, n, fit);
        return;
    }
    fit->mesor = beta[0];
    fit->a = beta[1];
    fit->b = beta[2];
    fit->n = n;
    fit->flat = 0;
}

double tone_fit_predict(const ToneFit *fit, double hour) {
    if (fit->flat) return fit->mesor;
    double w = TONE_TWO_PI_OVER_24 * hour;
    return fit->mesor + fit->a * cos(w) + fit->b * sin(w);
}

/* ---------------------------------------------------------------- Step 4 */

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

double tone_median(double *scratch, int n) {
    /* SPEC GAP 3. "MAD" implies a median but not a convention for even counts.
     * Chosen: the mean of the two central order statistics, which is what
     * NumPy, R and Swift's usual formulations all do. Taking the lower of the
     * two would shift sigma by a fraction of a percent -- enough to fail 1e-6,
     * not enough to notice by eye. */
    if (n <= 0) return NAN;
    qsort(scratch, (size_t)n, sizeof(double), cmp_double);
    if (n % 2) return scratch[n / 2];
    return 0.5 * (scratch[n / 2 - 1] + scratch[n / 2]);
}

double tone_robust_sigma(double *scratch, int n, double mad_scale) {
    if (n < 2) return NAN;
    double *copy = malloc((size_t)n * sizeof(double));
    if (!copy) return NAN;
    memcpy(copy, scratch, (size_t)n * sizeof(double));
    double med = tone_median(copy, n);
    for (int i = 0; i < n; i++) copy[i] = fabs(scratch[i] - med);
    double mad = tone_median(copy, n);
    free(copy);

    double sigma = mad_scale * mad;
    if (sigma <= TONE_MIN_SIGMA) {
        /* SPEC GAP 4. "fall back to the sample SD" does not say whether the
         * divisor is n or n-1. Chosen: n-1, because "sample SD" conventionally
         * means the unbiased estimator, and because the same wording elsewhere
         * in the spec (the daily sigma_S) is unambiguously a sample SD. */
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += scratch[i];
        double mean = sum / (double)n;
        double ss = 0.0;
        for (int i = 0; i < n; i++) { double d = scratch[i] - mean; ss += d * d; }
        sigma = sqrt(ss / (double)(n - 1));
    }
    return (sigma > TONE_MIN_SIGMA) ? sigma : NAN;
}

/* ---------------------------------------------------------------- Step 5 */

void tone_weights(const ToneConfig *cfg, double *w_x, double *w_h) {
    if (cfg->sigma2_eps_x > 0.0 && cfg->sigma2_eps_h > 0.0) {
        *w_x = cfg->lambda_hrv / cfg->sigma2_eps_x;
        *w_h = 1.0 / cfg->sigma2_eps_h;
    } else {
        /* SPEC GAP 5. The spec leaves the weights as TODO and forbids
         * substituting a plausible number, but says nothing about what the
         * engine should do when they are absent -- and the engine has to do
         * something to run against the example fixture. Chosen: equal weights
         * (lambda, 1), matching "an unmeasured weight is the thing this
         * ordering exists to prevent" -- equal weights are visibly a
         * placeholder, where 0.7/0.3 would look like a decision. */
        *w_x = cfg->lambda_hrv;
        *w_h = 1.0;
    }
}

double tone_combine(double z_x, double z_h, const ToneConfig *cfg) {
    double w_x, w_h;
    tone_weights(cfg, &w_x, &w_h);
    double total = w_x + w_h;
    if (total <= 0.0 || !isfinite(z_x) || !isfinite(z_h)) return NAN;
    return (w_x * (-z_x) + w_h * z_h) / total;
}

/* ---------------------------------------------------------------- Step 6 */

static const double TONE_T_CRIT_975[31] = {
    0.0,
    12.706205, 4.302653, 3.182446, 2.776445, 2.570582,
    2.446912,  2.364624, 2.306004, 2.262157, 2.228139,
    2.200985,  2.178813, 2.160369, 2.144787, 2.131450,
    2.119905,  2.109816, 2.100922, 2.093024, 2.085963,
    2.079614,  2.073873, 2.068658, 2.063899, 2.059539,
    2.055529,  2.051831, 2.048407, 2.045230, 2.042272
};
static const double TONE_Z_CRIT_975 = 1.959964;

double tone_t_crit(int df, int use_t) {
    if (!use_t) return TONE_Z_CRIT_975;
    if (df <= 0) return NAN;
    if (df <= 30) return TONE_T_CRIT_975[df];
    return TONE_Z_CRIT_975;
}

/* ------------------------------------------------------------- pipeline */

typedef struct {
    double *hours;
    double *values;
    double *resid;
} Channel;

static void score_channel(const double *hours, const double *values,
                          int index, int lo, const ToneConfig *cfg,
                          double *resid_out, double *sigma_out, double *z_out,
                          int *flat_out, double *scratch) {
    int n = index - lo;
    ToneFit fit;
    if (n < cfg->min_baseline_windows) {
        flat_fit(values + lo, n, &fit);
    } else {
        tone_cosinor_fit(hours + lo, values + lo, n, cfg->min_baseline_hours, &fit);
    }
    *flat_out = fit.flat;
    *resid_out = values[index] - tone_fit_predict(&fit, hours[index]);
    for (int i = 0; i < n; i++)
        scratch[i] = values[lo + i] - tone_fit_predict(&fit, hours[lo + i]);
    double sigma = tone_robust_sigma(scratch, n, cfg->mad_scale);
    *sigma_out = sigma;
    *z_out = (isfinite(sigma) && sigma > 0.0) ? *resid_out / sigma : NAN;
}

int tone_score_windows(const ToneWindow *windows, int n, const ToneConfig *cfg,
                       ToneScore *out) {
    /* SPEC GAP 6. "Over the trailing 28 days, excluding the window being
     * scored" leaves the interval's endpoints open. Chosen: a window j is in
     * the baseline of window i when epoch_j >= epoch_i - 28*86400 AND j < i by
     * position. Half-open at the old end, strictly-earlier at the new end, and
     * positional rather than by-time so that two windows sharing a timestamp
     * cannot each sit in the other's baseline. */
    if (n <= 0) return 0;

    double *hours = calloc((size_t)n, sizeof(double));
    double *xs = calloc((size_t)n, sizeof(double));
    double *hs = calloc((size_t)n, sizeof(double));
    double *epochs = calloc((size_t)n, sizeof(double));
    double *scratch = calloc((size_t)n, sizeof(double));
    int *map = calloc((size_t)n, sizeof(int));
    if (!hours || !xs || !hs || !epochs || !scratch || !map) {
        free(hours); free(xs); free(hs); free(epochs); free(scratch); free(map);
        return 1;
    }

    int u = 0;
    for (int i = 0; i < n; i++) {
        memset(&out[i], 0, sizeof(ToneScore));
        out[i].x = out[i].h = NAN;
        out[i].resid_x = out[i].resid_h = NAN;
        out[i].sigma_x = out[i].sigma_h = NAN;
        out[i].z_x = out[i].z_h = NAN;
        out[i].s = NAN;

        ToneMetrics m = tone_window_metrics(windows[i].rr, windows[i].n, cfg);
        if (!m.usable) continue;
        out[i].x = log(m.rmssd);
        out[i].h = log(m.mean_hr);
        hours[u] = windows[i].hour;
        xs[u] = out[i].x;
        hs[u] = out[i].h;
        epochs[u] = windows[i].epoch;
        map[u] = i;
        u++;
    }

    double span = cfg->baseline_days * 86400.0;
    for (int k = 0; k < u; k++) {
        ToneScore *ws = &out[map[k]];
        int lo = 0;
        while (lo < k && epochs[lo] < epochs[k] - span) lo++;
        ws->n_baseline = k - lo;
        if (ws->n_baseline < cfg->min_scoring_windows) continue;

        int flat_x = 0, flat_h = 0;
        score_channel(hours, xs, k, lo, cfg, &ws->resid_x, &ws->sigma_x, &ws->z_x,
                      &flat_x, scratch);
        score_channel(hours, hs, k, lo, cfg, &ws->resid_h, &ws->sigma_h, &ws->z_h,
                      &flat_h, scratch);
        ws->flat_baseline = flat_x || flat_h;
        ws->s = tone_combine(ws->z_x, ws->z_h, cfg);
        ws->scored = isfinite(ws->s);
    }

    free(hours); free(xs); free(hs); free(epochs); free(scratch); free(map);
    return 0;
}

int tone_daily_scores(const ToneWindow *windows, const ToneScore *scores, int n,
                      const ToneConfig *cfg, ToneDaily *out) {
    /* SPEC GAP 7. "A day" is never defined. Chosen: the local calendar date of
     * the window's start, consistent with Step 3 using local clock hour -- a
     * 1 a.m. window belongs to the night you went to bed on, in your own
     * timezone, not to whatever date it was in UTC.
     *
     * SPEC GAP 8. "borrows the pooled within-day SD from other days" does not
     * define the pooling. Chosen: the classical pooled estimate,
     * sqrt(sum_d SS_d / sum_d (n_d - 1)) over days with n_d >= 2, with df =
     * sum_d (n_d - 1) for the t critical value on such a day. */
    int count = 0;
    double pooled_ss = 0.0;
    int pooled_df = 0;

    /* Days are contiguous in a time-ordered input, so one linear pass. */
    int i = 0;
    while (i < n) {
        if (!scores[i].scored) { i++; continue; }
        int day = windows[i].day;
        double sum = 0.0;
        int members = 0;
        for (int j = i; j < n; j++) {
            if (windows[j].day != day) break;
            if (scores[j].scored) { sum += scores[j].s; members++; }
        }
        if (members >= 2) {
            double mean = sum / (double)members;
            for (int j = i; j < n; j++) {
                if (windows[j].day != day) break;
                if (scores[j].scored) { double d = scores[j].s - mean; pooled_ss += d * d; }
            }
            pooled_df += members - 1;
        }
        while (i < n && windows[i].day == day) i++;
    }
    double pooled_sd = (pooled_df > 0) ? sqrt(pooled_ss / (double)pooled_df) : NAN;

    i = 0;
    while (i < n) {
        if (!scores[i].scored) { i++; continue; }
        int day = windows[i].day;
        double sum = 0.0;
        int members = 0;
        int end = i;
        for (int j = i; j < n; j++) {
            if (windows[j].day != day) break;
            end = j + 1;
            if (scores[j].scored) { sum += scores[j].s; members++; }
        }
        double mean = sum / (double)members;
        double sd, se, crit;
        int pooled;
        if (members >= 2) {
            double ss = 0.0;
            for (int j = i; j < end; j++)
                if (scores[j].scored) { double d = scores[j].s - mean; ss += d * d; }
            sd = sqrt(ss / (double)(members - 1));
            se = sd / sqrt((double)members);
            crit = tone_t_crit(members - 1, cfg->interval_is_t);
            pooled = 0;
        } else {
            sd = pooled_sd;
            se = pooled_sd;
            crit = tone_t_crit(pooled_df > 1 ? pooled_df : 1, cfg->interval_is_t);
            pooled = 1;
        }
        out[count].day = day;
        out[count].mean = mean;
        out[count].n = members;
        out[count].sd = sd;
        out[count].se = se;
        out[count].sd_pooled = pooled;
        if (isfinite(se) && isfinite(crit)) {
            out[count].lo = mean - crit * se;
            out[count].hi = mean + crit * se;
        } else {
            out[count].lo = out[count].hi = NAN;
        }
        count++;
        i = end;
    }
    return count;
}
