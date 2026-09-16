/* Fixture runner: the parity gate of watch/CLAUDE.md section 4, in C.
 *
 *     ./tone_parity ../watch/fixtures.example.json
 *
 * Feeds every window in the file to the engine in order, then asserts each
 * `"checked": true` window's expected fields and every daily aggregate to
 * within the file's own tolerance. Exit 0 on parity, 1 on any mismatch.
 */

#include "engine.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Howard Hinnant's days_from_civil: calendar date -> days since 1970-01-01. */
static long days_from_civil(int y, int m, int d) {
    y -= m <= 2;
    long era = (y >= 0 ? y : y - 399) / 400;
    unsigned yoe = (unsigned)(y - era * 400);
    unsigned doy = (unsigned)((153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1);
    unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097L + (long)doe - 719468L;
}

/* Parse "YYYY-MM-DDTHH:MM:SS[.frac][Z|+HH:MM|-HH:MM]".
 * Writes UTC epoch seconds, the LOCAL fractional clock hour, and the LOCAL
 * calendar day. Returns 0 on failure. */
static int parse_iso(const char *s, double *epoch, double *hour, int *day) {
    int Y, M, D, hh, mm;
    double ss;
    if (sscanf(s, "%4d-%2d-%2dT%2d:%2d:%lf", &Y, &M, &D, &hh, &mm, &ss) != 6) return 0;

    /* Find the UTC offset by scanning BACKWARDS from the end. Scanning forwards
     * from a fixed index is how the first version of this function silently
     * failed: for a whole-second timestamp the offset sign sits at index 19,
     * and a guard meant to skip the date's hyphens skipped it too. Every epoch
     * then came out shifted by a constant -- invisible to the baseline window,
     * which only uses differences, and visible only in the day grouping. The
     * mutation suite caught it; nothing else would have.
     *
     * The date's own hyphens are at indices 4 and 7 and 'T' is at 10, so
     * refusing to look at or before index 10 cannot confuse them for a sign. */
    long offset = 0;
    size_t len = strlen(s);
    const char *tz = NULL;
    if (len > 0 && (s[len - 1] == 'Z' || s[len - 1] == 'z')) {
        tz = s + len - 1;  /* explicit UTC */
    } else {
        for (size_t i = len; i-- > 11; ) {
            if (s[i] == '+' || s[i] == '-') { tz = s + i; break; }
        }
    }
    if (!tz) {
        /* A naive timestamp would make "local clock hour" and "local day"
         * guesses. Refuse rather than guess: this is the failure that hid. */
        return 0;
    }
    if (*tz == '+' || *tz == '-') {
        int oh = 0, om = 0;
        if (sscanf(tz + 1, "%2d:%2d", &oh, &om) != 2) return 0;
        offset = (long)(oh * 3600 + om * 60) * (*tz == '-' ? -1 : 1);
    }

    long days = days_from_civil(Y, M, D);
    *epoch = (double)(days * 86400L) + hh * 3600.0 + mm * 60.0 + ss - (double)offset;
    *hour = hh + mm / 60.0 + ss / 3600.0;
    *day = (int)days;  /* the LOCAL date, because Y-M-D was read as local */
    return 1;
}

static int day_from_string(const char *s) {
    int Y, M, D;
    if (sscanf(s, "%4d-%2d-%2d", &Y, &M, &D) != 3) return -999999;
    return (int)days_from_civil(Y, M, D);
}

static int failures = 0;
static const int MAX_REPORTED = 20;

static void check(const char *what, const char *where, double got,
                  const JsonValue *want, double tol) {
    int ok;
    if (json_is_null(want)) {
        ok = !isfinite(got);
    } else if (want->type != JSON_NUMBER) {
        ok = 0;
    } else {
        double w = want->number;
        ok = isfinite(got) && fabs(got - w) <= tol * (fabs(w) > 1.0 ? fabs(w) : 1.0);
    }
    if (!ok) {
        if (failures < MAX_REPORTED) {
            if (json_is_null(want))
                fprintf(stderr, "  %s %s: got %.17g, expected null/non-finite\n", where, what, got);
            else if (want->type != JSON_NUMBER)
                fprintf(stderr, "  %s %s: expected value is not a number\n", where, what);
            else
                fprintf(stderr, "  %s %s: got %.17g, expected %.17g (diff %.3g)\n",
                        where, what, got, want->number, fabs(got - want->number));
        }
        failures++;
    }
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "../watch/fixtures.example.json";

    char *text = json_read_file(path);
    if (!text) { fprintf(stderr, "cannot read %s\n", path); return 2; }
    const char *err = NULL;
    JsonValue *doc = json_parse(text, &err);
    free(text);
    if (!doc) { fprintf(stderr, "cannot parse %s: %s\n", path, err); return 2; }

    double tol = json_num(doc, "tolerance", 1e-6);
    const JsonValue *cfg_json = json_get(doc, "config");
    ToneConfig cfg;
    tone_config_defaults(&cfg);
    if (cfg_json) {
        cfg.artifact_threshold = json_num(cfg_json, "artifact_threshold", cfg.artifact_threshold);
        cfg.min_intervals = (int)json_num(cfg_json, "min_intervals", cfg.min_intervals);
        cfg.baseline_days = json_num(cfg_json, "baseline_days", cfg.baseline_days);
        cfg.min_baseline_windows = (int)json_num(cfg_json, "min_baseline_windows", cfg.min_baseline_windows);
        cfg.min_baseline_hours = (int)json_num(cfg_json, "min_baseline_hours", cfg.min_baseline_hours);
        cfg.min_scoring_windows = (int)json_num(cfg_json, "min_scoring_windows", cfg.min_scoring_windows);
        cfg.lambda_hrv = json_num(cfg_json, "lambda_hrv", cfg.lambda_hrv);
        cfg.sigma2_eps_x = json_num(cfg_json, "sigma2_eps_x", -1.0);
        cfg.sigma2_eps_h = json_num(cfg_json, "sigma2_eps_h", -1.0);
        cfg.mad_scale = json_num(cfg_json, "mad_scale", cfg.mad_scale);
        const char *interval = json_str(cfg_json, "interval", "t");
        cfg.interval_is_t = (strcmp(interval, "normal") != 0);
    }

    const JsonValue *wins = json_get(doc, "windows");
    if (!wins || wins->type != JSON_ARRAY) {
        fprintf(stderr, "fixture has no windows array\n");
        json_free(doc);
        return 2;
    }
    int n = (int)wins->count;

    ToneWindow *windows = calloc((size_t)n, sizeof(ToneWindow));
    double **rr_store = calloc((size_t)n, sizeof(double *));
    ToneScore *scores = calloc((size_t)n, sizeof(ToneScore));
    ToneDaily *daily = calloc((size_t)n, sizeof(ToneDaily));
    if (!windows || !rr_store || !scores || !daily) {
        fprintf(stderr, "out of memory\n");
        return 2;
    }

    for (int i = 0; i < n; i++) {
        const JsonValue *w = json_at(wins, (size_t)i);
        const JsonValue *rr = json_get(w, "rr_ms");
        int m = (rr && rr->type == JSON_ARRAY) ? (int)rr->count : 0;
        double *buf = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
        for (int k = 0; k < m; k++) {
            const JsonValue *v = json_at(rr, (size_t)k);
            buf[k] = (v && v->type == JSON_NUMBER) ? v->number : NAN;
        }
        rr_store[i] = buf;
        windows[i].rr = buf;
        windows[i].n = m;
        windows[i].on_demand = json_bool(w, "on_demand", 0);

        const char *start = json_str(w, "start", NULL);
        if (!start || !parse_iso(start, &windows[i].epoch, &windows[i].hour, &windows[i].day)) {
            fprintf(stderr, "window %d: cannot parse start timestamp\n", i);
            return 2;
        }
        /* Cross-check our own clock-hour derivation against the fixture's, which
         * catches a UTC-vs-local mistake before it becomes a baseline mistake. */
        const JsonValue *hour_json = json_get(w, "hour");
        if (hour_json && hour_json->type == JSON_NUMBER
            && fabs(hour_json->number - windows[i].hour) > 1e-6) {
            fprintf(stderr, "window %d: clock hour %.9g disagrees with fixture %.9g\n",
                    i, windows[i].hour, hour_json->number);
            failures++;
        }
    }

    if (tone_score_windows(windows, n, &cfg, scores) != 0) {
        fprintf(stderr, "scoring failed (allocation)\n");
        return 2;
    }
    int n_daily = tone_daily_scores(windows, scores, n, &cfg, daily);

    int checked = 0;
    for (int i = 0; i < n; i++) {
        const JsonValue *w = json_at(wins, (size_t)i);
        if (!json_bool(w, "checked", 0)) continue;
        const JsonValue *e = json_get(w, "expected");
        if (!e) continue;
        checked++;

        char where[64];
        snprintf(where, sizeof(where), "window %d", i);

        ToneMetrics m = tone_window_metrics(windows[i].rr, windows[i].n, &cfg);
        check("n_kept", where, (double)m.n_kept, json_get(e, "n_kept"), tol);
        check("rmssd", where, m.rmssd, json_get(e, "rmssd"), tol);
        check("mean_hr", where, m.mean_hr, json_get(e, "mean_hr"), tol);

        const ToneScore *s = &scores[i];
        check("x", where, s->x, json_get(e, "x"), tol);
        check("h", where, s->h, json_get(e, "h"), tol);
        check("resid_x", where, s->resid_x, json_get(e, "resid_x"), tol);
        check("resid_h", where, s->resid_h, json_get(e, "resid_h"), tol);
        check("sigma_x", where, s->sigma_x, json_get(e, "sigma_x"), tol);
        check("sigma_h", where, s->sigma_h, json_get(e, "sigma_h"), tol);
        check("z_x", where, s->z_x, json_get(e, "z_x"), tol);
        check("z_h", where, s->z_h, json_get(e, "z_h"), tol);
        check("s", where, s->s, json_get(e, "s"), tol);
    }

    const JsonValue *days = json_get(doc, "daily");
    int days_checked = 0;
    if (days && days->type == JSON_ARRAY) {
        for (size_t d = 0; d < days->count; d++) {
            const JsonValue *want = json_at(days, d);
            const char *day_str = json_str(want, "day", "");
            int key = day_from_string(day_str);
            const ToneDaily *got = NULL;
            for (int k = 0; k < n_daily; k++)
                if (daily[k].day == key) { got = &daily[k]; break; }
            char where[64];
            snprintf(where, sizeof(where), "daily %s", day_str);
            if (!got) {
                if (failures < MAX_REPORTED) fprintf(stderr, "  %s: missing\n", where);
                failures++;
                continue;
            }
            days_checked++;
            check("mean", where, got->mean, json_get(want, "mean"), tol);
            check("n", where, (double)got->n, json_get(want, "n"), tol);
            check("sd", where, got->sd, json_get(want, "sd"), tol);
            check("se", where, got->se, json_get(want, "se"), tol);
            check("lo", where, got->lo, json_get(want, "lo"), tol);
            check("hi", where, got->hi, json_get(want, "hi"), tol);
        }
    }

    printf("fixture:   %s\n", path);
    printf("windows:   %d fed, %d checked\n", n, checked);
    printf("daily:     %d checked\n", days_checked);
    printf("tolerance: %g relative\n", tol);
    if (failures == 0) {
        printf("PARITY: pass\n");
    } else {
        if (failures > MAX_REPORTED)
            fprintf(stderr, "  ... and %d more\n", failures - MAX_REPORTED);
        printf("PARITY: FAIL (%d mismatches)\n", failures);
    }

    for (int i = 0; i < n; i++) free(rr_store[i]);
    free(rr_store);
    free(windows);
    free(scores);
    free(daily);
    json_free(doc);
    return failures == 0 ? 0 : 1;
}
