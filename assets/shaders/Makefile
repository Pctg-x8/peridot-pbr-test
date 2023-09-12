
TARGETS = $(patsubst %.csh,%.pvp,$(wildcard *.csh)) $(patsubst %.csh,%.pvp,$(wildcard **/*.csh))

.SUFFIXES: .pvp .csh
.PHONY: all
all: $(TARGETS)
.csh.pvp:
	../../../peridot/tools/target/release/peridot-shaderbuild $<

