build/%.frag.spv: cpp/%.frag
CSV_VIEWER_TARGET := csv_viewer
CSV_COMPARE_TARGET := compare_csv
BUILD_DIR  := build
NPROC      := $(shell nproc)
CUDA_BIN   := /usr/local/cuda/bin
CUDA_LIB   := /usr/local/cuda/lib64

export PATH := $(CUDA_BIN):$(PATH)
export LD_LIBRARY_PATH := $(CUDA_LIB):$(LD_LIBRARY_PATH)

# Detect GLSL → SPIR-V compiler (glslc preferred, glslangValidator as fallback)
GLSLC         := $(shell which glslc 2>/dev/null)
GLSLANGVAL    := $(shell which glslangValidator 2>/dev/null)

ifdef GLSLC
  SHADER_CMD  = $(GLSLC) $< -o $@
else ifdef GLSLANGVAL
  SHADER_CMD  = $(GLSLANGVAL) -V $< -o $@
else
  SHADER_CMD  = $(error Neither glslc nor glslangValidator found. \
    Install with: sudo apt install glslang-tools)
endif

SHADER_SRCS := cpp/hit_point.vert cpp/hit_point.frag \
			   cpp/mesh_wire.vert cpp/mesh_wire.frag \
			   cpp/cone.frag
SHADER_SPVS := $(patsubst cpp/%.vert,build/%.vert.spv,$(filter %.vert,$(SHADER_SRCS))) \
			   $(patsubst cpp/%.frag,build/%.frag.spv,$(filter %.frag,$(SHADER_SRCS)))

.PHONY: make run run-csv-viewer run-csv-compare bench-setup clean

# Ensure build directory exists before compiling shaders
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Default build target: running 'make' builds everything (shaders + cmake build)
$(SHADER_SPVS): | $(BUILD_DIR)

all: $(SHADER_SPVS)
	@if [ -n "$(OPTIX_ROOT)" ]; then \
	  cmake -B $(BUILD_DIR) -DOptiX_INSTALL_DIR=$(OPTIX_ROOT) -DCMAKE_CUDA_COMPILER=nvcc; \
	else \
	  cmake -B $(BUILD_DIR) -DCMAKE_CUDA_COMPILER=nvcc; \
	fi
	cmake --build $(BUILD_DIR) -j$(NPROC)

.DEFAULT_GOAL := all

build/%.vert.spv: cpp/%.vert
	$(SHADER_CMD)

build/%.frag.spv: cpp/%.frag
	$(SHADER_CMD)

bench-setup:
	@echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
	@echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo > /dev/null 2>&1 || \
	 echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null 2>&1 || true

run: all bench-setup
	taskset -c 0 ./$(BUILD_DIR)/grca

run-csv-viewer:
	@if [ -z "$(filter-out $@,$(MAKECMDGOALS))" ]; then \
		echo "Usage: make run-csv-viewer <csv_file>"; \
		exit 1; \
	fi
	./$(BUILD_DIR)/$(CSV_VIEWER_TARGET) $(filter-out $@,$(MAKECMDGOALS))

run-csv-compare:
	@opts=""; files=""; \
	if [ -n "$(tol)" ]; then opts="$$opts --tol $(tol)"; fi; \
	if [ -n "$(match)" ]; then opts="$$opts --match $(match)"; fi; \
	for arg in $(filter-out $@,$(MAKECMDGOALS)); do \
	  case "$$arg" in \
	    --tol*|-t*|--match*|-m*) opts="$$opts $$arg" ;; \
	    *.csv) files="$$files $$arg" ;; \
	    *) files="$$files $$arg" ;; \
	  esac; \
	done; \
	file_count=$(shell echo $$files | wc -w); \
	if [ "$$(echo $$files | wc -w)" -lt 2 ]; then \
	  echo "Usage: make run-csv-compare [tol=<tolerance>] [match=<min-match%>] [--tol <tolerance>] [--match <min-match%>] <file1.csv> <file2.csv> [file3.csv ...]"; \
	  exit 1; \
	fi; \
	./$(BUILD_DIR)/$(CSV_COMPARE_TARGET) $$opts $$files

clean:
	rm -rf $(BUILD_DIR)/*

# Catch-all: silently ignore extra args passed to run-csv-viewer
%:
	@:
