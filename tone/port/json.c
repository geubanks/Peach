#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *p;
    const char *err;
} Parser;

static JsonValue *parse_value(Parser *ps);

static JsonValue *alloc_value(JsonType type) {
    JsonValue *v = calloc(1, sizeof(JsonValue));
    if (v) v->type = type;
    return v;
}

static void skip_ws(Parser *ps) {
    while (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' || *ps->p == '\r') ps->p++;
}

static char *parse_string_raw(Parser *ps) {
    if (*ps->p != '"') { ps->err = "expected string"; return NULL; }
    ps->p++;
    size_t cap = 32, len = 0;
    char *out = malloc(cap);
    if (!out) { ps->err = "out of memory"; return NULL; }
    while (*ps->p && *ps->p != '"') {
        char c = *ps->p++;
        if (c == '\\') {
            char e = *ps->p++;
            switch (e) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'u': {
                    /* Fixtures are ASCII; anything else becomes '?'. */
                    for (int i = 0; i < 4 && *ps->p; i++) ps->p++;
                    c = '?';
                    break;
                }
                default: c = e; break;  /* covers \" \\ \/ */
            }
            if (!e) { ps->err = "unterminated escape"; free(out); return NULL; }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *grown = realloc(out, cap);
            if (!grown) { ps->err = "out of memory"; free(out); return NULL; }
            out = grown;
        }
        out[len++] = c;
    }
    if (*ps->p != '"') { ps->err = "unterminated string"; free(out); return NULL; }
    ps->p++;
    out[len] = '\0';
    return out;
}

static JsonValue *parse_array(Parser *ps) {
    JsonValue *v = alloc_value(JSON_ARRAY);
    if (!v) { ps->err = "out of memory"; return NULL; }
    ps->p++;  /* '[' */
    skip_ws(ps);
    if (*ps->p == ']') { ps->p++; return v; }
    size_t cap = 8;
    v->items = malloc(cap * sizeof(JsonValue *));
    if (!v->items) { ps->err = "out of memory"; json_free(v); return NULL; }
    for (;;) {
        skip_ws(ps);
        JsonValue *item = parse_value(ps);
        if (!item) { json_free(v); return NULL; }
        if (v->count == cap) {
            cap *= 2;
            JsonValue **grown = realloc(v->items, cap * sizeof(JsonValue *));
            if (!grown) { ps->err = "out of memory"; json_free(item); json_free(v); return NULL; }
            v->items = grown;
        }
        v->items[v->count++] = item;
        skip_ws(ps);
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == ']') { ps->p++; return v; }
        ps->err = "expected , or ] in array";
        json_free(v);
        return NULL;
    }
}

static JsonValue *parse_object(Parser *ps) {
    JsonValue *v = alloc_value(JSON_OBJECT);
    if (!v) { ps->err = "out of memory"; return NULL; }
    ps->p++;  /* '{' */
    skip_ws(ps);
    if (*ps->p == '}') { ps->p++; return v; }
    size_t cap = 8;
    v->keys = malloc(cap * sizeof(char *));
    v->values = malloc(cap * sizeof(JsonValue *));
    if (!v->keys || !v->values) { ps->err = "out of memory"; json_free(v); return NULL; }
    for (;;) {
        skip_ws(ps);
        char *key = parse_string_raw(ps);
        if (!key) { json_free(v); return NULL; }
        skip_ws(ps);
        if (*ps->p != ':') { ps->err = "expected :"; free(key); json_free(v); return NULL; }
        ps->p++;
        skip_ws(ps);
        JsonValue *val = parse_value(ps);
        if (!val) { free(key); json_free(v); return NULL; }
        if (v->nkeys == cap) {
            cap *= 2;
            char **gk = realloc(v->keys, cap * sizeof(char *));
            JsonValue **gv = realloc(v->values, cap * sizeof(JsonValue *));
            if (!gk || !gv) {
                ps->err = "out of memory";
                if (gk) v->keys = gk;
                if (gv) v->values = gv;
                free(key); json_free(val); json_free(v);
                return NULL;
            }
            v->keys = gk;
            v->values = gv;
        }
        v->keys[v->nkeys] = key;
        v->values[v->nkeys] = val;
        v->nkeys++;
        skip_ws(ps);
        if (*ps->p == ',') { ps->p++; continue; }
        if (*ps->p == '}') { ps->p++; return v; }
        ps->err = "expected , or } in object";
        json_free(v);
        return NULL;
    }
}

static JsonValue *parse_value(Parser *ps) {
    skip_ws(ps);
    switch (*ps->p) {
        case '{': return parse_object(ps);
        case '[': return parse_array(ps);
        case '"': {
            JsonValue *v = alloc_value(JSON_STRING);
            if (!v) { ps->err = "out of memory"; return NULL; }
            v->string = parse_string_raw(ps);
            if (!v->string) { json_free(v); return NULL; }
            return v;
        }
        case 't':
            if (strncmp(ps->p, "true", 4) != 0) { ps->err = "bad literal"; return NULL; }
            ps->p += 4;
            { JsonValue *v = alloc_value(JSON_BOOL); if (v) v->boolean = 1; else ps->err = "oom"; return v; }
        case 'f':
            if (strncmp(ps->p, "false", 5) != 0) { ps->err = "bad literal"; return NULL; }
            ps->p += 5;
            { JsonValue *v = alloc_value(JSON_BOOL); if (!v) ps->err = "oom"; return v; }
        case 'n':
            if (strncmp(ps->p, "null", 4) != 0) { ps->err = "bad literal"; return NULL; }
            ps->p += 4;
            { JsonValue *v = alloc_value(JSON_NULL); if (!v) ps->err = "oom"; return v; }
        default: {
            char *end = NULL;
            double d = strtod(ps->p, &end);
            if (end == ps->p) { ps->err = "expected a value"; return NULL; }
            ps->p = end;
            JsonValue *v = alloc_value(JSON_NUMBER);
            if (!v) { ps->err = "out of memory"; return NULL; }
            v->number = d;
            return v;
        }
    }
}

JsonValue *json_parse(const char *text, const char **err) {
    Parser ps = { text, NULL };
    JsonValue *v = parse_value(&ps);
    if (v) {
        skip_ws(&ps);
        if (*ps.p != '\0') {
            json_free(v);
            v = NULL;
            ps.err = "trailing content after value";
        }
    }
    if (err) *err = ps.err ? ps.err : "ok";
    return v;
}

void json_free(JsonValue *v) {
    if (!v) return;
    free(v->string);
    for (size_t i = 0; i < v->count; i++) json_free(v->items[i]);
    free(v->items);
    for (size_t i = 0; i < v->nkeys; i++) {
        free(v->keys[i]);
        json_free(v->values[i]);
    }
    free(v->keys);
    free(v->values);
    free(v);
}

char *json_read_file(const char *path) {
    FILE *fh = fopen(path, "rb");
    if (!fh) return NULL;
    if (fseek(fh, 0, SEEK_END) != 0) { fclose(fh); return NULL; }
    long size = ftell(fh);
    if (size < 0) { fclose(fh); return NULL; }
    rewind(fh);
    char *buf = malloc((size_t)size + 1);
    if (!buf) { fclose(fh); return NULL; }
    size_t got = fread(buf, 1, (size_t)size, fh);
    fclose(fh);
    buf[got] = '\0';
    return buf;
}

const JsonValue *json_get(const JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->nkeys; i++)
        if (strcmp(obj->keys[i], key) == 0) return obj->values[i];
    return NULL;
}

double json_num(const JsonValue *obj, const char *key, double fallback) {
    const JsonValue *v = json_get(obj, key);
    return (v && v->type == JSON_NUMBER) ? v->number : fallback;
}

int json_bool(const JsonValue *obj, const char *key, int fallback) {
    const JsonValue *v = json_get(obj, key);
    return (v && v->type == JSON_BOOL) ? v->boolean : fallback;
}

const char *json_str(const JsonValue *obj, const char *key, const char *fallback) {
    const JsonValue *v = json_get(obj, key);
    return (v && v->type == JSON_STRING) ? v->string : fallback;
}

const JsonValue *json_at(const JsonValue *arr, size_t i) {
    if (!arr || arr->type != JSON_ARRAY || i >= arr->count) return NULL;
    return arr->items[i];
}

int json_is_null(const JsonValue *v) {
    return !v || v->type == JSON_NULL;
}
