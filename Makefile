BUILD_DIR = build
INCLUDE_DIR = include

SRC_DIR = src
OBJ_DIR = obj
SRCS := $(shell find $(SRC_DIR) -type f -name '*.c')
OBJS := $(SRCS:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

LIB_DIR = lib
LIB_SRCS = 

# Conditional compilation of JSON-related libraries `yyjson` and `nibble_base64`.
#	`make LITE3_JSON_LIBRARIES=1` --> JSON enabled (default)
#	`make LITE3_JSON_LIBRARIES=0` --> JSON disabled
# This feature requires `malloc()`.
# Disable this if you are in a no-malloc environment, or you do not need to convert to/from JSON.
# This prevents bloating your build target and can be useful in embedded or constrained environments.
LITE3_JSON_LIBRARIES ?= 1
ifeq ($(LITE3_JSON_LIBRARIES),1)
	LIB_SRCS += $(shell find $(LIB_DIR)/yyjson -name '*.c')
	LIB_SRCS += $(shell find $(LIB_DIR)/nibble_base64 -name '*.c')
endif

LIB_OBJS := $(LIB_SRCS:$(LIB_DIR)/%.c=$(OBJ_DIR)/%.o)

TESTS_DIR = tests
TESTS_SRCS := $(shell find $(TESTS_DIR) -type f -name '*.c')
TESTS_OBJS := $(TESTS_SRCS:$(TESTS_DIR)/%.c=$(OBJ_DIR)/tests/%.o)
TESTS_EXES := $(TESTS_SRCS:$(TESTS_DIR)/%.c=$(BUILD_DIR)/tests/%)

EXAMPLES_DIR = examples
EXAMPLES_SRCS := $(shell find $(EXAMPLES_DIR) -type f -name '*.c')
EXAMPLES_OBJS := $(EXAMPLES_SRCS:$(EXAMPLES_DIR)/%.c=$(OBJ_DIR)/examples/%.o)
EXAMPLES_EXES := $(EXAMPLES_SRCS:$(EXAMPLES_DIR)/%.c=$(BUILD_DIR)/examples/%)


# executable name
EXE = lite3
STATIC_LIB = lib$(EXE).a

# compiler
CC = gcc
# linker
LD = gcc
# archive utility
AR = ar
ARFLAGS = rcs


CFLAGS = -Wall -Wformat=2 -Wextra -Wconversion -Wstrict-aliasing=3 -Wcast-align -Wshadow
# For finer-grained dead code elimination use the following CFLAGS, but only if the linker of your master project uses -Wl,--gc-sections
# Source: https://gcc.gnu.org/onlinedocs/gnat_ugn/Compilation-options.html
# CFLAGS += -ffunction-sections -fdata-sections

# pre-processor flags
CPPFLAGS = -I$(INCLUDE_DIR) -I$(LIB_DIR)
# dependency-generation flags
DEPFLAGS = -MMD -MP
# linker flags
LDFLAGS = 
# library flags
LDLIBS = 

CFLAGS_DEBUG = -Og -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3 -fstack-protector-all
# CFLAGS_DEBUG += -Winline

CFLAGS_RELEASE = -O2 -DNDEBUG
# CFLAGS_RELEASE += -march=native -mtune=native -Wa,-mbranches-within-32B-boundaries
# CFLAGS_RELEASE += -march=x86-64-v3 -mtune=generic -Wa,-mbranches-within-32B-boundaries
# CFLAGS_RELEASE += -march=skylake -mtune=skylake -Wa,-mbranches-within-32B-boundaries
# CFLAGS_RELEASE += -march=znver3 -mtune=znver3 -Wa,-mbranches-within-32B-boundaries


COMPILE.c = $(CC) $(DEPFLAGS) $(CFLAGS) $(CPPFLAGS) -c -o $@


.PRECIOUS: $(OBJ_DIR)/%.o

$(OBJ_DIR)/%.o:		$(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(COMPILE.c) $<

$(OBJ_DIR)/%.o:		$(LIB_DIR)/%.c
	@mkdir -p $(dir $@)
	$(COMPILE.c) $<

$(OBJ_DIR)/tests/%.o:	$(TESTS_DIR)/%.c
	@mkdir -p $(dir $@)
	$(COMPILE.c) $<

$(OBJ_DIR)/examples/%.o:$(EXAMPLES_DIR)/%.c
	@mkdir -p $(dir $@)
	$(COMPILE.c) $<


LINK.o = $(LD) $(LDFLAGS) $(LDLIBS) $^ -o $@

# link objects into executables
$(BUILD_DIR)/tests/%: $(OBJ_DIR)/tests/%.o $(OBJS) $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(LINK.o)

$(BUILD_DIR)/examples/%: $(OBJ_DIR)/examples/%.o $(OBJS) $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(LINK.o)

LOCALIZED_SYMS = $(OBJ_DIR)/localized_syms.txt
SYMBOL_PREFIX = lite3_internal_

$(LOCALIZED_SYMS): $(LIB_OBJS)
	@rm -f $@
	@mkdir -p $(dir $@)
	@touch $@
	@for obj in $^; do \
		nm --defined-only -g $$obj | grep ' [A-TV-Z] ' | awk '{print $$3 " $(SYMBOL_PREFIX)" $$3}' >> $@; \
	done

$(BUILD_DIR)/$(STATIC_LIB): $(LOCALIZED_SYMS) $(OBJS) $(LIB_OBJS)
	@if [ -s $(LOCALIZED_SYMS) ]; then \
		for file in $(OBJS) $(LIB_OBJS); do \
			objcopy --redefine-syms=$(LOCALIZED_SYMS) $$file $$file.tmp && mv $$file.tmp $$file; \
		done; \
	fi
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $(OBJS) $(LIB_OBJS)
	@rm -f $(LOCALIZED_SYMS)


.DEFAULT_GOAL = all
.PHONY: all
all: CFLAGS += $(CFLAGS_RELEASE)
all: $(BUILD_DIR)/$(STATIC_LIB)

.PHONY: examples
examples: CFLAGS += $(CFLAGS_DEBUG)
examples: $(EXAMPLES_EXES)

.PHONY: tests
tests: CFLAGS += $(CFLAGS_DEBUG)
tests: $(TESTS_EXES)
	@echo "\033[1;34m========= Running Tests =========\033[0m"
	@set -e; \
	PASS=0; FAIL=0; \
	for test in $(TESTS_EXES); do \
		echo -n "\033[1;33m[RUN] $$(basename $$test)\033[0m ... "; \
		if $$test; then \
			echo "\033[1;32mPASS\033[0m"; \
			PASS=$$((PASS+1)); \
		else \
			echo "\033[1;31mFAIL\033[0m"; \
			FAIL=$$((FAIL+1)); \
		fi; \
	done; \
	echo "\033[1;34m============ Results ============\033[0m"; \
	echo "\033[1;32m   [+] $$PASS passed\033[0m   \033[1;31m[-] $$FAIL failed\033[0m"; \
	echo "\033[1;34m=================================\033[0m"; \
	if [ $$FAIL -ne 0 ]; then exit 1; fi

.PHONY: help
help:
	@echo "Available targets:"
	@echo "    all        - Build the static library with -O2 optimizations (default)"
	@echo "    tests      - Build and run all tests"
	@echo "    examples   - Build all examples"
	@echo "    install    - Install library in \`/usr/local\` (for pkg-config)"
	@echo "    uninstall  - Uninstall library"
	@echo "    clean      - Remove all build artifacts"
	@echo "    help       - Show this help message"
	@echo ""
	@echo "Optional features (set via \`make VARIABLE=value\`):"
	@echo "    LITE3_JSON_LIBRARIES - Conditional compilation of JSON-related libraries \`yyjson\` and \`nibble_base64\` (default: 1)"
	@echo "            make LITE3_JSON_LIBRARIES=1  -> Enable"
	@echo "            make LITE3_JSON_LIBRARIES=0  -> Disable, minimal build"
	@echo "    This feature requires \`malloc()\`."
	@echo "    Disable this if you are in a no-malloc environment, or you do not need to convert to/from JSON."

PREFIX ?= /usr/local

PC_INCLUDE_DIR = $(DESTDIR)$(PREFIX)/include
PC_LIB_DIR = $(DESTDIR)$(PREFIX)/lib
PC_PKGCONFIG_DIR = $(DESTDIR)$(PREFIX)/lib/pkgconfig

.PHONY: install
install: $(BUILD_DIR)/$(STATIC_LIB)
	install -d $(PC_INCLUDE_DIR)
	install -d $(PC_LIB_DIR)
	install -d $(PC_PKGCONFIG_DIR)
	install -m 644 $(INCLUDE_DIR)/lite3.h $(PC_INCLUDE_DIR)
	install -m 644 $(INCLUDE_DIR)/lite3_context_api.h $(PC_INCLUDE_DIR)
	install -m 644 $(BUILD_DIR)/$(STATIC_LIB) $(PC_LIB_DIR)
	sed -e 's|^prefix=.*|prefix=$(PREFIX)|' \
	    -e 's|^exec_prefix=.*|exec_prefix=$${prefix}|' \
	    -e 's|^includedir=.*|includedir=$${prefix}/include|' \
	    -e 's|^libdir=.*|libdir=$${exec_prefix}/lib|' \
	    lite3.pc > $(PC_PKGCONFIG_DIR)/lite3.pc
	@echo "Library installed successfully"
	@echo ""
	@echo "Check library version:"
	@echo "	pkg-config --modversion lite3"
	@echo "Validate the lite3.pc file:"
	@echo "	pkg-config --validate lite3 && echo \"lite3.pc is valid\""
	@echo "Example compilation:"
	@echo '	gcc -o program program.c $$(pkg-config --libs --cflags --static lite3)'
	@echo ""

.PHONY: uninstall
uninstall:
	$(RM) $(PC_INCLUDE_DIR)/lite3.h
	$(RM) $(PC_INCLUDE_DIR)/lite3_context_api.h
	$(RM) $(PC_LIB_DIR)/$(STATIC_LIB)
	$(RM) $(PC_PKGCONFIG_DIR)/lite3.pc

.PHONY: clean
clean:
	$(RM) -r $(OBJ_DIR)
	$(RM) -r $(BUILD_DIR)


# include compiler-generated dependency rules
-include $(OBJS:.o=.d) $(LIB_OBJS:.o=.d) $(TESTS_OBJS:.o=.d) $(EXAMPLES_OBJS:.o=.d)