/* Minimal JSON reader -- just enough to read a Tone fixture file.
 *
 * Not a general-purpose library: no streaming, no big-number care, no \u
 * beyond the BMP-to-'?' fallback. It exists because the fixture runner has to
 * be dependency-free to be worth anything as a portability demonstration.
 */
#ifndef TONE_JSON_H
#define TONE_JSON_H

#include <stddef.h>

typedef enum {
    JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT
} JsonType;

typedef struct JsonValue JsonValue;

struct JsonValue {
    JsonType type;
    int boolean;
    double number;
    char *string;
    JsonValue **items;   /* array */
    size_t count;
    char **keys;         /* object */
    JsonValue **values;
    size_t nkeys;
};

/* Parse NUL-terminated text. Returns NULL on malformed input and writes a
 * short reason into `err` (may be NULL). Caller owns the result. */
JsonValue *json_parse(const char *text, const char **err);
void json_free(JsonValue *v);

/* Read a whole file into a malloc'd NUL-terminated buffer. NULL on failure. */
char *json_read_file(const char *path);

/* Accessors. Missing keys and type mismatches return the fallback. */
const JsonValue *json_get(const JsonValue *obj, const char *key);
double json_num(const JsonValue *obj, const char *key, double fallback);
int json_bool(const JsonValue *obj, const char *key, int fallback);
const char *json_str(const JsonValue *obj, const char *key, const char *fallback);
const JsonValue *json_at(const JsonValue *arr, size_t i);

/* True when the value is JSON null, which is how the fixture writes a
 * non-finite expectation (Swift's JSONDecoder rejects bare NaN). */
int json_is_null(const JsonValue *v);

#endif
