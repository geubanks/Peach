/* Tone ScoreEngine -- a second implementation, in C99 + libm and nothing else.
 *
 * Why this exists
 * ---------------
 * The watch app's ScoreEngine is meant to be a port of the Python, verified by
 * fixtures. This file is that port, written in C rather than Swift because no
 * Swift toolchain was reachable -- and written *from `../watch/CLAUDE.md`
 * alone*, without reading the Python, precisely so that the exercise tests
 * something worth testing: whether the spec is complete enough to implement
 * from. Phase 3.1's "done when" is that a second reader can build the app from
 * the spec. This is that second reader, in a language where nothing can be
 * hand-waved.
 *
 * Every place the spec did not determine an answer is marked `SPEC GAP` in
 * engine.c. Those are the file's real output. Swift transcribes almost
 * one-for-one from here: scalar loops, no linear algebra, no allocation inside
 * the maths.
 */
#ifndef TONE_ENGINE_H
#define TONE_ENGINE_H

#include <stddef.h>

#define TONE_TWO_PI_OVER_24 0.26179938779914943654
#define TONE_PIVOT_EPS 1e-9
#define TONE_MIN_SIGMA 1e-9

typedef struct {
    double artifact_threshold;   /* 0.20 */
    int    min_intervals;        /* 30 */
    double baseline_days;        /* 28 */
    int    min_baseline_windows; /* 30 */
    int    min_baseline_hours;   /* 5  */
    int    min_scoring_windows;  /* 30 */
    double lambda_hrv;           /* 1.0 */
    double sigma2_eps_x;         /* <= 0 means "not measured" */
    double sigma2_eps_h;
    double mad_scale;            /* 1.4826 */
    int    interval_is_t;        /* 1 for Student's t, 0 for 1.96 */
} ToneConfig;

void tone_config_defaults(ToneConfig *cfg);

typedef struct {
    const double *rr;  /* inter-beat intervals, ms, in order */
    int n;
    double epoch;      /* seconds since the Unix epoch (UTC) */
    double hour;       /* local clock hour in [0, 24), fractional */
    int day;           /* local calendar day, as days from 1970-01-01 */
    int on_demand;
} ToneWindow;

typedef struct {
    int usable;
    int n_kept;
    int n_diffs;
    double rmssd;
    double mean_hr;
} ToneMetrics;

typedef struct {
    double mesor, a, b;
    int n;
    int flat;
} ToneFit;

typedef struct {
    int scored;
    int n_baseline;
    int flat_baseline;
    double x, h;
    double resid_x, resid_h;
    double sigma_x, sigma_h;
    double z_x, z_h;
    double s;
} ToneScore;

typedef struct {
    int day;
    double mean;
    int n;
    double sd, se, lo, hi;
    int sd_pooled;
} ToneDaily;

/* Step 1 */
void   tone_filter_rr(const double *rr, int n, double threshold, unsigned char *keep);
double tone_rmssd(const double *rr, int n, const unsigned char *keep, int *n_diffs);
double tone_mean_rr(const double *rr, int n, const unsigned char *keep);
ToneMetrics tone_window_metrics(const double *rr, int n, const ToneConfig *cfg);

/* Step 3 */
int  tone_solve3(const double ata[9], const double atb[3], double out[3]);
void tone_cosinor_fit(const double *hours, const double *y, int n,
                      int min_distinct_hours, ToneFit *fit);
double tone_fit_predict(const ToneFit *fit, double hour);

/* Step 4 */
double tone_median(double *scratch, int n);     /* sorts `scratch` in place */
double tone_robust_sigma(double *scratch, int n, double mad_scale);

/* Step 5 */
void   tone_weights(const ToneConfig *cfg, double *w_x, double *w_h);
double tone_combine(double z_x, double z_h, const ToneConfig *cfg);

/* Step 6 */
double tone_t_crit(int df, int use_t);

/* Whole pipeline. `windows` must be sorted by epoch ascending. `out` must have
 * room for `n` scores. Returns 0 on success, non-zero if allocation failed. */
int tone_score_windows(const ToneWindow *windows, int n, const ToneConfig *cfg,
                       ToneScore *out);

/* Daily aggregation over the scored windows. Writes at most `n` entries into
 * `out` and returns how many were written, or -1 on allocation failure. */
int tone_daily_scores(const ToneWindow *windows, const ToneScore *scores, int n,
                      const ToneConfig *cfg, ToneDaily *out);

#endif
