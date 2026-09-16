CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic
LDLIBS  := -lm
FIXTURE ?= ../watch/fixtures.example.json

tone_parity: runner.c engine.c json.c engine.h json.h
	$(CC) $(CFLAGS) -o $@ runner.c engine.c json.c $(LDLIBS)

.PHONY: check clean
check: tone_parity
	./tone_parity $(FIXTURE)

clean:
	rm -f tone_parity
