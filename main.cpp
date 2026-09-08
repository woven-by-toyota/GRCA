// =============================================================================
// main.cpp  –  Unified GRCA entry point
//
// Loads config.json, parses the scene and test batches, dispatches to each
// backend in turn, writes per-batch CSV output, and prints timing breakdowns.
//
// Config format (preferred):
//   { "scenes": ["scenes/default.json"], "active_tests": ["tests/rebuild.json", ...] }
//
// Flat config (backward-compatible):
//   { "backend": "grca-cpu", "robots": [...], ... }
//
// Usage:
//   grca [config.json] [--backend grca-cpu|grca-cuda|optix|optix-crti|embree|tinybvh-cpu|tinybvh-gpu|hybrid-cpu|hybrid-gpu|all|all_t|all_h|cpu|cpu_t|cpu_h|gpu|gpu_t|gpu_h|gpu-all|bvh|bvh_t|grca|hybrid]
//                     [--out <prefix>] [--debug] [--refit] [--rebuild]
//                     [--active tests[N] | tests[N]/batches[M]]
//
// Output:
//   benchmark/<YYYYMMDD_HHMMSS>-<user>-<host>/batch_N/<type>_<backend>_<ts>.csv
// =============================================================================

#define TINYOBJLOADER_IMPLEMENTATION
#include "third_party/tiny_obj_loader.h"

#include "include/grca_common.h"
#include "cpp/cpu.h"

#ifdef HAVE_CUDA
#  include <cuda_runtime_api.h>
#  include "cuda/grca_cuda.h"
#endif
#ifdef HAVE_OPTIX
#  include "optix_cuda/optix.h"
#endif
#ifdef HAVE_EMBREE
#  include "cpp/embree.h"
#endif
#ifdef HAVE_TINYBVH
#  include "cpp/tinybvh_cpu.h"
#  include "cpp/tinybvh_gpu.h"
#endif
#ifdef HAVE_EMBREE
#  include "cpp/hybrid_cpu.h"
#endif
#if defined(HAVE_OPTIX) && defined(HAVE_CUDA)
#  include "cpp/hybrid_gpu.h"
#endif

#include "cpp/hit_viewer.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <pwd.h>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <sys/utsname.h>
#include <unistd.h>

using json = nlohmann::json;
using clk  = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Directory / path helpers
// ---------------------------------------------------------------------------

static void ensure_dir(const std::string& path) {
    std::filesystem::create_directories(path);
}

static std::string sanitize_path_component(const std::string& s) {
    std::string out = s;
    for (char& c : out)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '-';
    return out;
}

static std::string get_username() {
    const char* user = getenv("USER");
    if (user) return user;
    struct passwd* pw = getpwuid(getuid());
    return pw ? pw->pw_name : "unknown";
}

static std::string get_hostname() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) return buf;
    struct utsname uts;
    if (uname(&uts) == 0) return uts.nodename;
    return "unknown";
}

// ---------------------------------------------------------------------------
// Fast RNG for per-triangle chaos — xoshiro128+ (~3 cycles/output vs mt19937's ~7)
// ---------------------------------------------------------------------------
struct XoShiro128Plus {
    uint32_t s[4] = {};
    void seed(uint64_t v) {
        // splitmix64 to fill all four state words
        for (int i = 0; i < 4; ++i) {
            v += 0x9e3779b97f4a7c15ULL;
            uint64_t z = v;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z ^= z >> 31;
            s[i] = (uint32_t)z;
        }
        if (!s[0] && !s[1] && !s[2] && !s[3]) s[0] = 1;
    }
    uint32_t next() {
        const uint32_t r = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t;
        s[3] = (s[3] << 11) | (s[3] >> 21);
        return r;
    }
    float f01()  { return (next() >> 8) * (1.f / 16777216.f); }   // [0, 1)
    float f11()  { return f01() * 2.f - 1.f; }                    // [-1, 1)
    float frange(float lo, float w) { return lo + f01() * w; }    // [lo, lo+w)
};

static inline float fast_rsqrt(float x) {
    return 1.f / sqrtf(x);
}

// Expand group aliases to compiled-in backend lists.
//   all    — GRCA-CPU + GRCA-CUDA + OptiX + Embree  (no TinyBVH, no Hybrid)
//   cpu    — GRCA-CPU + Embree                      (no TinyBVH, no Hybrid)
//   gpu    — GRCA-CUDA + OptiX                      (no TinyBVH, no Hybrid)
//   all_t  — everything in "all" + TinyBVH-CPU + TinyBVH-GPU
//   cpu_t  — GRCA-CPU + Embree + TinyBVH-CPU
//   gpu_t  — GRCA-CUDA + OptiX + TinyBVH-GPU
//   bvh    — Embree + OptiX
//   bvh_t  — Embree + OptiX + TinyBVH-CPU + TinyBVH-GPU
//   grca    — GRCA-CPU + GRCA-CUDA
//   gpu-all — GRCA-CUDA + OptiX + OptiX-CRTI
//   all_h  — everything in "all" + Hybrid-CPU + Hybrid-GPU
//   cpu_h  — everything in "cpu" + Hybrid-CPU
//   gpu_h  — everything in "gpu" + Hybrid-GPU
//   hybrid — Hybrid-CPU + Hybrid-GPU
static std::vector<std::string> expand_backends(const std::string& backend) {
    if (backend != "all"   && backend != "cpu"   && backend != "gpu"   &&
        backend != "all_t" && backend != "cpu_t" && backend != "gpu_t" &&
        backend != "bvh"   && backend != "bvh_t" && backend != "grca"   &&
        backend != "gpu-all" &&
        backend != "all_h" && backend != "cpu_h" && backend != "gpu_h" &&
        backend != "hybrid")
        return {backend};
    std::vector<std::string> out;
    bool want_cpu        = (backend == "all"   || backend == "cpu"   || backend == "all_t" || backend == "cpu_t" ||
                            backend == "all_h" || backend == "cpu_h");
    bool want_gpu        = (backend == "all"   || backend == "gpu"   || backend == "all_t" || backend == "gpu_t" ||
                            backend == "all_h" || backend == "gpu_h");
    bool want_tbvh_cpu   = (backend == "all_t" || backend == "cpu_t" || backend == "bvh_t");
    bool want_tbvh_gpu   = (backend == "all_t" || backend == "gpu_t" || backend == "bvh_t");
    bool want_bvh        = (backend == "bvh"   || backend == "bvh_t");
    bool want_grca        = (backend == "grca");
    bool want_gpu_all    = (backend == "gpu-all");
    bool want_hybrid_cpu = (backend == "all_h" || backend == "cpu_h" || backend == "hybrid");
    bool want_hybrid_gpu = (backend == "all_h" || backend == "gpu_h" || backend == "hybrid");
#ifdef HAVE_OPTIX
    if (want_gpu || want_bvh || want_gpu_all)
        out.push_back("optix");
    if (want_gpu_all)
        out.push_back("optix-crti");
#endif
#ifdef HAVE_CUDA
    if (want_gpu || want_grca || want_gpu_all)
        out.push_back("grca-cuda");
#endif
#if defined(HAVE_OPTIX) && defined(HAVE_CUDA)
    if (want_hybrid_gpu)
        out.push_back("hybrid-gpu");
#endif
#ifdef HAVE_TINYBVH
    if (want_tbvh_gpu)
        out.push_back("tinybvh-gpu");
#endif
#ifdef HAVE_EMBREE
    if (want_cpu || want_bvh)
        out.push_back("embree");
#endif
#ifdef HAVE_TINYBVH
    if (want_tbvh_cpu)
        out.push_back("tinybvh-cpu");
#endif
    if (want_cpu || want_grca)
        out.push_back("grca-cpu");
#ifdef HAVE_EMBREE
    if (want_hybrid_cpu)
        out.push_back("hybrid-cpu");
#endif
    return out;
}

// ---------------------------------------------------------------------------
// System-info helpers (for CSV header)
// ---------------------------------------------------------------------------

static std::string read_line_from_file(const char* path, const char* pattern) {
    std::ifstream file(path);
    std::string line;
    std::regex re(pattern);
    while (std::getline(file, line))
        if (std::regex_search(line, re)) return line;
    return "";
}

static std::string get_cpu_model() {
    std::string line = read_line_from_file("/proc/cpuinfo", R"(model name\s*:)");
    auto pos = line.find(':');
    return pos != std::string::npos ? line.substr(pos + 2) : "Unknown";
}

static std::string get_ram_gib() {
    std::string line = read_line_from_file("/proc/meminfo", R"(MemTotal)");
    std::smatch m;
    if (std::regex_search(line, m, std::regex(R"(\d+)"))) {
        double gib = std::stol(m[0]) / (1024.0 * 1024.0);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << gib << " GiB";
        return oss.str();
    }
    return "Unknown";
}

// Returns current process resident set size in MB (Linux only).
static double get_process_ram_mb() {
    std::string line = read_line_from_file("/proc/self/status", "VmRSS");
    std::smatch m;
    if (std::regex_search(line, m, std::regex(R"(\d+)")))
        return std::stol(m[0]) / 1024.0;
    return 0.0;
}

static std::string get_gpu_model() {
    FILE* p = popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1", "r");
    if (!p) return "Unknown";
    char buf[256]; std::string r;
    if (fgets(buf, sizeof(buf), p)) { r = buf; r.erase(r.find_last_not_of(" \n\r\t") + 1); }
    pclose(p);
    return r.empty() ? "Unknown" : r;
}

static std::string get_gpu_vram() {
    FILE* p = popen("nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -n1", "r");
    if (!p) return "Unknown";
    char buf[256]; std::string r;
    if (fgets(buf, sizeof(buf), p)) { r = buf; r.erase(r.find_last_not_of(" \n\r\t") + 1); }
    pclose(p);
    return r.empty() ? "Unknown" : r;
}

// ---------------------------------------------------------------------------
// Backend label (display name)
// ---------------------------------------------------------------------------

static std::string backendLabel(const RunConfig& cfg) {
    if (cfg.backend == "grca-cuda") return "GRCA_cuda";
    if (cfg.backend == "grca-cpu")  return "GRCA_cpu_vec" + std::to_string(cfg.grca_cpu_packet_size);
    if (cfg.backend == "optix" || cfg.backend == "optix-crti") {
        int n = cfg.optix_rebuild_after_frames;
        std::string mode = (n > 0) ? "hb" + std::to_string(n) : (n == 0 ? "rb" : "rf");
        std::string comp = cfg.optix_compaction ? "cmp_" : "";
        std::string is   = (cfg.backend == "optix-crti") ? "crti_" : "";
        return "Optix_" + is + comp + mode;
    }
    if (cfg.backend == "embree") {
        int n = cfg.embree_rebuild_after_frames;
        std::string mode = (n > 0) ? "hb" + std::to_string(n) : (n == 0 ? "rb" : "rf");
        std::string comp = cfg.embree_compaction ? "cmp_" : "";
        return "Embree_vec" + std::to_string(cfg.embree_packet_size) + "_" + comp + mode;
    }
    if (cfg.backend == "tinybvh-cpu") {
        int n = cfg.tinybvh_cpu_rebuild_after_frames;
        std::string mode = (n > 0) ? "hb" + std::to_string(n) : (n == 0 ? "rb" : "rf");
        return "TinyBVH_cpu_" + mode;
    }
    if (cfg.backend == "tinybvh-gpu") {
        int n = cfg.tinybvh_gpu_rebuild_after_frames;
        std::string mode = (n > 0) ? "hb" + std::to_string(n) : (n == 0 ? "rb" : "rf");
        return "TinyBVH_gpu_" + mode;
    }
    if (cfg.backend == "hybrid-cpu") {
        // Embree BVH is always forced to refit mode (-1) for static geometry.
        return "Hybrid_cpu_Embree_vec" + std::to_string(cfg.embree_packet_size) + "_rf_GRCA_vec" + std::to_string(cfg.grca_cpu_packet_size);
    }
    if (cfg.backend == "hybrid-gpu") {
        // OptiX AS is always forced to refit mode (-1) for static geometry.
        return "Hybrid_gpu_Optix_rf_GRCA_cuda";
    }
    return cfg.backend;
}

static std::string backendApiLabel(const RunConfig& cfg) {
    if (cfg.backend == "grca-cuda")     return "GRCA_cuda";
    if (cfg.backend == "grca-cpu")      return "GRCA_cpu";
    if (cfg.backend == "optix")        return "Optix";
    if (cfg.backend == "optix-crti")   return "Optix_crti";
    if (cfg.backend == "embree")       return "Embree";
    if (cfg.backend == "tinybvh-cpu")  return "TinyBVH_cpu";
    if (cfg.backend == "tinybvh-gpu")  return "TinyBVH_gpu";
    if (cfg.backend == "hybrid-cpu")   return "Hybrid_cpu";
    if (cfg.backend == "hybrid-gpu")   return "Hybrid_gpu";
    return cfg.backend;
}

// ---------------------------------------------------------------------------
// CSV filename builder
// ---------------------------------------------------------------------------

static std::string make_timestamped_csv_name(const std::string& type,
                                              const RunConfig& cfg,
                                              const std::string& out_dir = ".") {
    std::string prefix = type + "_" + backendLabel(cfg) + "_";
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&now_c, &tm_buf);
    std::ostringstream oss;
    oss << out_dir << "/" << prefix << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".csv";
    return oss.str();
}

// ---------------------------------------------------------------------------
// Batch result collection and summary
// ---------------------------------------------------------------------------

// Find the compare_csv binary alongside this executable
static std::string find_compare_csv_bin() {
    char buf[4096] = {};
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf)-1);
    if (n > 0) {
        std::string exe(buf, n);
        auto pos = exe.find_last_of('/');
        if (pos != std::string::npos) {
            std::string candidate = exe.substr(0, pos) + "/compare_csv";
            if (access(candidate.c_str(), X_OK) == 0) return candidate;
        }
    }
    return "";
}

struct CompareStats {
    bool   ran            = false;
    double tol            = 0.0;
    double min_match_crit = 0.0;
    int    pairs_matched  = 0;
    int    pairs_total    = 0;
    double min_match_pct  = 100.0;
};

// Shell out to compare_csv for a group of files, printing its output to stderr.
// Returns aggregate statistics parsed from the output.
static CompareStats run_compare_csv(const char* title,
                             const std::vector<std::string>& files,
                             double tol, double match_pct)
{
    CompareStats stats;
    stats.tol            = tol;
    stats.min_match_crit = match_pct;
    if (files.size() < 2) return stats;
    static std::string bin = find_compare_csv_bin();
    fprintf(stderr, "\n  %s  (tol=%.4g  min-match=%.4g%%)\n", title, tol, match_pct);
    if (bin.empty()) {
        fprintf(stderr, "  [compare_csv binary not found — skipping]\n");
        return stats;
    }
    stats.ran = true;
    std::ostringstream cmd;
    cmd << "\"" << bin << "\" --simple --tol " << tol << " --match " << match_pct;
    for (const auto& f : files) cmd << " \"" << f << "\"";
    cmd << " 2>&1";
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) { fprintf(stderr, "  [compare_csv] popen failed\n"); return stats; }
    char rbuf[4096];
    while (fgets(rbuf, sizeof(rbuf), pipe)) {
        fprintf(stderr, "  %s", rbuf);
        std::string line(rbuf);
        // Summary line: "N/M pairs match  (tol=T  min-match=P%)"
        {
            int pm = 0, pt = 0;
            if (sscanf(line.c_str(), " %d/%d pairs match", &pm, &pt) == 2) {
                stats.pairs_matched = pm;
                stats.pairs_total   = pt;
                continue;
            }
        }
        // Per-pair lines contain uppercase "MATCH" or "MISMATCH"; extract min match%.
        // Format examples: "MATCH   dist=99.76%(max=...)"  or  "MATCH   all=100.00%"
        if (line.find("MATCH") != std::string::npos) {
            const char* p = line.c_str();
            while (*p) {
                if (*p == '=') {
                    char* endp = nullptr;
                    double v = std::strtod(p + 1, &endp);
                    if (endp && endp > p + 1 && *endp == '%')
                        stats.min_match_pct = std::min(stats.min_match_pct, v);
                }
                ++p;
            }
        }
    }
    pclose(pipe);
    return stats;
}

// Run plot_results.py on a results CSV path (or directory).
// If run_idx >= 0, passes --run <run_idx> to only generate that run's overview.
static void run_plot_results(const std::string& path, int run_idx = -1, bool session_only = false)
{
    // plot_results.py lives at <exe_dir>/../benchmark/plot_results.py
    char buf[4096] = {};
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf)-1);
    std::string script;
    if (n > 0) {
        std::string exe(buf, n);
        auto pos = exe.find_last_of('/');
        if (pos != std::string::npos)
            script = exe.substr(0, pos) + "/../benchmark/plot_results.py";
    }
    if (script.empty() || access(script.c_str(), R_OK) != 0)
        script = "benchmark/plot_results.py";  // fallback: relative to cwd

    if (access(script.c_str(), R_OK) != 0) {
        fprintf(stderr, "\n  [plot_results] script not found — skipping plots\n");
        return;
    }

    std::ostringstream cmd;
    cmd << "python3 \"" << script << "\" \"" << path << "\"";
    if (run_idx >= 0)
        cmd << " --run " << run_idx;
    if (session_only)
        cmd << " --session-only";
    cmd << " 2>&1";
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) { fprintf(stderr, "  [plot_results] popen failed\n"); return; }
    char rbuf[4096];
    fprintf(stderr, "\n  Plots\n");
    while (fgets(rbuf, sizeof(rbuf), pipe)) fprintf(stderr, "  %s", rbuf);
    pclose(pipe);
}


struct BatchBackendResult {
    std::string backend;
    std::string label;
    std::string api_label;
    std::string hit_dist_csv;
    std::string pose_csv;
    double init_cpu_ms           = 0;
    double avg_fps               = 0;
    double avg_rtf               = 0;
    double avg_total_cpu_ms      = 0;
    double avg_lidar_sim_cpu     = 0;
    double min_lidar_sim_cpu     = 0;
    double max_lidar_sim_cpu     = 0;
    double avg_lidar_sim_gpu     = 0;
    double min_lidar_sim_gpu     = 0;
    double max_lidar_sim_gpu     = 0;
    double avg_trace_gpu         = 0;
    double avg_trace_cpu         = 0;
    double avg_dynamic_upload_ms = 0;
    double avg_bvh_rebuild_gpu_ms= 0;
    double avg_bvh_refit_gpu_ms  = 0;
    double avg_bvh_rebuild_cpu_ms= 0;
    double avg_bvh_refit_cpu_ms  = 0;
    double avg_readback_ms       = 0;
    double avg_sync_readback_ms  = 0;
    double avg_fill_hits_ms      = 0;
    int    perf_frames           = 0;
    int    batch_lsr             = 0;
};

static void print_batch_summary(size_t runIdx, size_t batchIdx,
                                 const std::string& results_csv_path,
                                 const std::vector<BatchBackendResult>& results,
                                 double tol = 0.001)
{
    if (results.empty()) return;

    const char* SEP = "  --------------------------------------------------------\n";
    fprintf(stderr, "\n%s", SEP);
    fprintf(stderr, "  Batch %zu Summary  (%zu backend%s)\n",
            batchIdx, results.size(), results.size() == 1 ? "" : "s");
    fprintf(stderr, "%s", SEP);

    // CPU ranking
    std::vector<size_t> cpu_ord(results.size());
    std::iota(cpu_ord.begin(), cpu_ord.end(), 0);
    std::sort(cpu_ord.begin(), cpu_ord.end(), [&](size_t a, size_t b) {
        return results[a].avg_total_cpu_ms < results[b].avg_total_cpu_ms;
    });
    fprintf(stderr, "\n  CPU  (avg total, lower is better)\n");
    fprintf(stderr, "  %-14s  %10s  %12s  %10s\n",
            "Backend", "Total(ms)", "LidarSim(ms)", "Trace(ms)");
    fprintf(stderr, "  %-14s  %10s  %12s  %10s\n",
            "-------", "---------", "------------", "---------");
    for (size_t i = 0; i < cpu_ord.size(); ++i) {
        const auto& r = results[cpu_ord[i]];
        fprintf(stderr, "  %zu. %-12s  %9.3f  %12.3f  %9.3f\n",
                i+1, r.label.c_str(),
                r.avg_total_cpu_ms, r.avg_lidar_sim_cpu, r.avg_trace_cpu);
    }

    // GPU ranking (only if any backend has GPU data)
    bool any_gpu = false;
    for (const auto& r : results) if (r.avg_lidar_sim_gpu > 0.0) { any_gpu = true; break; }
    if (any_gpu) {
        std::vector<size_t> gpu_ord(results.size());
        std::iota(gpu_ord.begin(), gpu_ord.end(), 0);
        std::sort(gpu_ord.begin(), gpu_ord.end(), [&](size_t a, size_t b) {
            double ga = results[a].avg_lidar_sim_gpu > 0 ? results[a].avg_lidar_sim_gpu : 1e9;
            double gb = results[b].avg_lidar_sim_gpu > 0 ? results[b].avg_lidar_sim_gpu : 1e9;
            return ga < gb;
        });
        fprintf(stderr, "\n  GPU  (avg total, lower is better)\n");
        fprintf(stderr, "  %-14s  %10s  %12s  %10s\n",
                "Backend", "Total(ms)", "LidarSim(ms)", "Trace(ms)");
        fprintf(stderr, "  %-14s  %10s  %12s  %10s\n",
                "-------", "---------", "------------", "---------");
        for (size_t i = 0; i < gpu_ord.size(); ++i) {
            const auto& r = results[gpu_ord[i]];
            if (r.avg_lidar_sim_gpu > 0) {
                fprintf(stderr, "  %zu. %-12s  %9.3f  %12.3f  %9.3f\n",
                        i+1, r.label.c_str(),
                        r.avg_lidar_sim_gpu, r.avg_lidar_sim_gpu, r.avg_trace_gpu);
            } else {
                fprintf(stderr, "  %zu. %-12s  %9s   %12s   %9s\n",
                        i+1, r.label.c_str(), "N/A", "N/A", "N/A");
            }
        }
    }

    // CSV match checks using compare_csv binary (only meaningful with 2+ backends)
    CompareStats hd_stats, pose_stats;
    {
        std::vector<std::string> hd_files, pose_files;
        for (const auto& r : results) {
            if (!r.hit_dist_csv.empty()) hd_files.push_back(r.hit_dist_csv);
            if (!r.pose_csv.empty())     pose_files.push_back(r.pose_csv);
        }
        if (results.size() >= 2) {
            hd_stats   = run_compare_csv("Hit Distance Check", hd_files,   tol, 95.0);
            pose_stats = run_compare_csv("Pose Check",         pose_files, tol, 95.0);
        }
        for (const auto& f : hd_files)   std::filesystem::remove(f);
        for (const auto& f : pose_files) std::filesystem::remove(f);
    }

    fprintf(stderr, "%s\n", SEP);

    // Write results CSV to session root
        std::ofstream rf(results_csv_path);
        rf << "run,batch,backend,api_label,perf_frames,lidar_sim_rate,"
            "avg_lidar_sim_cpu_ms,min_lidar_sim_cpu_ms,max_lidar_sim_cpu_ms,"
            "avg_dynamic_upload_ms,"
            "avg_bvh_rebuild_cpu_ms,avg_bvh_refit_cpu_ms,"
            "avg_trace_cpu_ms,"
            "avg_sync_readback_ms,"
            "avg_readback_ms,"
            "avg_fill_hits_ms,"
            "avg_lidar_sim_gpu_ms,min_lidar_sim_gpu_ms,max_lidar_sim_gpu_ms,"
            "avg_bvh_rebuild_gpu_ms,avg_bvh_refit_gpu_ms,"
            "avg_trace_gpu_ms,"
            "init_cpu_ms,"
            "avg_fps,avg_rtf,"
            "avg_total_cpu_ms,"
            "cmp_tol,cmp_min_match,"
            "hit_dist_pairs_matched,hit_dist_pairs_total,hit_dist_min_pct,"
            "pose_pairs_matched,pose_pairs_total,pose_min_pct\n";
        rf << std::fixed << std::setprecision(4);
        for (const auto& r : results) {
          rf << runIdx << "," << batchIdx << "," << r.label << "," << r.api_label << "," << r.perf_frames << ","
             << r.batch_lsr << ","
             << r.avg_lidar_sim_cpu      << "," << r.min_lidar_sim_cpu      << "," << r.max_lidar_sim_cpu << ","
             << r.avg_dynamic_upload_ms  << ","
             << r.avg_bvh_rebuild_cpu_ms << "," << r.avg_bvh_refit_cpu_ms  << ","
             << r.avg_trace_cpu          << ","
             << r.avg_sync_readback_ms   << ","
             << r.avg_readback_ms        << ","
             << r.avg_fill_hits_ms       << ","
             << r.avg_lidar_sim_gpu      << "," << r.min_lidar_sim_gpu      << "," << r.max_lidar_sim_gpu << ","
             << r.avg_bvh_rebuild_gpu_ms << "," << r.avg_bvh_refit_gpu_ms  << ","
             << r.avg_trace_gpu          << ","
             << r.init_cpu_ms            << ","
             << r.avg_fps                << "," << r.avg_rtf                << ","
             << r.avg_total_cpu_ms       << ","
             << hd_stats.tol            << "," << hd_stats.min_match_crit  << ","
             << hd_stats.pairs_matched  << "," << hd_stats.pairs_total     << "," << hd_stats.min_match_pct  << ","
             << pose_stats.pairs_matched << "," << pose_stats.pairs_total   << "," << pose_stats.min_match_pct << "\n";
        }
}

// ---------------------------------------------------------------------------
// Config/session summary writers
// ---------------------------------------------------------------------------


static void write_session_meta_csv(const std::string& path,
                                    const std::string& session_dir,
                                    const std::string& configPath,
                                    const json& j,
                                    bool is_new_format,
                                    const std::string& scene_path,
                                    const std::string& robots_path,
                                    const std::string& dyn_mesh_path,
                                    const RunConfig& scene_cfg,
                                    size_t staticTriCount, size_t dynTriCount,
                                    int totalRays)
{
    size_t lidarCount = 0;
    for (const auto& r : scene_cfg.robots) lidarCount += r.lidars.size();

    std::ofstream f(path);
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    f << "key,value\n";

    // --- Hardware ---
    f << "session,"  << session_dir << "\n";
    f << "date,"     << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << "\n";
    f << "os,Ubuntu 24.04.4 LTS\n";
    f << "cpu,"      << get_cpu_model() << "\n";
    f << "ram,"      << get_ram_gib()   << "\n";
    f << "gpu,"      << get_gpu_model() << "\n";
    f << "vram,"     << get_gpu_vram()  << "\n";

    // --- Config files used ---
    f << "config," << configPath << "\n";
    if (is_new_format) {
        f << "scene," << scene_path << "\n";
        f << "robot," << robots_path << "\n";
        if (!dyn_mesh_path.empty()) f << "dynamic_meshes," << dyn_mesh_path << "\n";
        if (j.contains("active_tests") && j["active_tests"].is_array())
            for (size_t i = 0; i < j["active_tests"].size(); ++i)
                f << "test_" << i << "," << j["active_tests"][i].get<std::string>() << "\n";
    }

    // --- Scene summary ---
    f << "static_mesh_count,"  << scene_cfg.static_meshes.size()  << "\n";
    f << "dynamic_mesh_count," << scene_cfg.dynamic_meshes.size() << "\n";
    f << "static_triangles,"   << staticTriCount                  << "\n";
    f << "dynamic_triangles,"  << dynTriCount                     << "\n";
    f << "total_triangles,"    << (staticTriCount + dynTriCount)   << "\n";
    f << "robot_count,"        << scene_cfg.robots.size()         << "\n";
    f << "lidar_count,"        << lidarCount                      << "\n";
    f << "total_rays,"         << totalRays                       << "\n";
}

// Copy a single file into dest_dir (flat, using its basename).
static void copy_file_to_session(const std::string& src, const std::string& dest_dir)
{
    std::error_code ec;
    namespace fs = std::filesystem;
    fs::path s(src);
    if (!fs::exists(s, ec)) {
        fprintf(stderr, "[warn] cannot find %s for copy\n", src.c_str());
        return;
    }
    fs::copy_file(s, fs::path(dest_dir) / s.filename(),
                  fs::copy_options::overwrite_existing, ec);
    if (ec)
        fprintf(stderr, "[warn] copy %s: %s\n", src.c_str(), ec.message().c_str());
}

// Copy a config file into session_dir, recreating its relative sub-path.
// e.g. rel="configs/tests/refit.json" → session_dir/configs/tests/refit.json
static void copy_config_file(const std::string& src_full, const std::string& rel_path,
                              const std::string& session_dir)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dest = fs::path(session_dir) / rel_path;
    fs::create_directories(dest.parent_path(), ec);
    fs::copy_file(fs::path(src_full), dest, fs::copy_options::overwrite_existing, ec);
    if (ec)
        fprintf(stderr, "[warn] copy %s: %s\n", src_full.c_str(), ec.message().c_str());
}

// ---------------------------------------------------------------------------
// Performance CSV writers
// ---------------------------------------------------------------------------

static void write_perf_csv_header(std::ofstream& out, float lidar_sim_rate) {
    out << "# lidar_sim_rate:," << lidar_sim_rate << "\n";
}

static void write_perf_csv_row(std::ostream& out, int frame, const FrameTimings& t) {
    out << frame                  << ","
        << t.ms_lidar_sim_cpu     << ","
        << t.ms_dynamic_upload    << ","
        << t.ms_bvh_rebuild_cpu   << ","
        << t.ms_bvh_refit_cpu     << ","
        << t.ms_trace_cpu         << ","
        << t.ms_sync_readback     << ","
        << t.ms_readback_cpu      << ","
        << t.ms_fillhits          << ","
        << t.ms_lidar_sim_gpu     << ","
        << t.ms_bvh_rebuild_gpu   << ","
        << t.ms_bvh_refit_gpu     << ","
        << t.ms_init_cpu          << ","
        << t.sim_frame_rate       << ","
        << t.real_time_factor     << ","
        << t.ms_total_cpu         << ","
        << t.mb_readback          << ","
        << t.hit_count            << "\n";
}

// ---------------------------------------------------------------------------
// Pose record frames config
// ---------------------------------------------------------------------------

struct PoseRecordFramesCfg {
    std::vector<std::pair<std::string, int>> candidates; // (type, index)
    int frames = -1;
};

static PoseRecordFramesCfg parsePoseRecordFrames(const json& j, const RunConfig& cfg) {
    PoseRecordFramesCfg out;
    if (j.contains("record_frames")) out.frames = j["record_frames"].get<int>();
    for (int idx = 0; idx < (int)cfg.robots.size(); ++idx)
        out.candidates.emplace_back("robot", idx);
    for (int idx = 0; idx < (int)cfg.dynamic_meshes.size(); ++idx)
        out.candidates.emplace_back("dynamic_mesh", idx);
    for (int idx = 0; idx < (int)cfg.static_meshes.size(); ++idx)
        out.candidates.emplace_back("static_mesh", idx);
    return out;
}

// ---------------------------------------------------------------------------
// Test configs
// ---------------------------------------------------------------------------

// Auto-detect the active test type from a batch JSON object.
static std::string getActiveBatchTest(const json& j) {
    if (j.contains("active_test") && j["active_test"].is_string())
        return j["active_test"].get<std::string>();
    if (j.contains("orbit_pairs_test"))  return "orbit_pairs_test";
    if (j.contains("lidar_orbit_test"))  return "lidar_orbit_test";
    if (j.contains("random_dynamic_test")) return "random_dynamic_test";
    return "";
}

struct OrbitPairsTestCfg {
    float orbit_radius                    = 2.0f;
    float orbit_speed                     = 1.0f;
    int   seed                            = 0;
    int   dyn_mesh_rot_interpolate_frames = -1;
    struct PairRef { int robot_idx = -1; int mesh_idx = -1; };
    std::vector<PairRef> pairs;
};

static std::optional<OrbitPairsTestCfg> parseOrbitPairsTest(const json& j,
                                                              const RunConfig& cfg) {
    if (!j.contains("orbit_pairs_test")) return std::nullopt;
    const auto& o = j["orbit_pairs_test"];
    OrbitPairsTestCfg tc;
    if (o.contains("orbit_radius")) tc.orbit_radius = o["orbit_radius"].get<float>();
    if (o.contains("orbit_speed"))  tc.orbit_speed  = o["orbit_speed"].get<float>();
    if (o.contains("seed"))         tc.seed         = o["seed"].get<int>();
    if (o.contains("dyn_mesh_rot_interpolate_frames"))
        tc.dyn_mesh_rot_interpolate_frames = o["dyn_mesh_rot_interpolate_frames"].get<int>();

    const std::string key = o.contains("pair_candidates") ? "pair_candidates" : "candidates";
    if (o.contains(key) && o[key].is_array()) {
        for (const auto& arr : o[key]) {
            if (!arr.is_array() || arr.size() != 2) continue;
            OrbitPairsTestCfg::PairRef ref;
            auto parse_idx = [](const std::string& s) -> int {
                size_t p0 = s.find('['), p1 = s.find(']');
                if (p0 == std::string::npos || p1 == std::string::npos) return -1;
                return std::stoi(s.substr(p0 + 1, p1 - p0 - 1));
            };
            if (arr[0].is_string()) {
                std::string s = arr[0].get<std::string>();
                if (s.find("robots[") == 0) ref.robot_idx = parse_idx(s);
            }
            if (arr[1].is_string()) {
                std::string s = arr[1].get<std::string>();
                if (s.find("dynamic_meshes[") == 0) ref.mesh_idx = parse_idx(s);
            }
            if (ref.robot_idx < 0 || ref.robot_idx >= (int)cfg.robots.size()) continue;
            if (ref.mesh_idx  < 0 || ref.mesh_idx  >= (int)cfg.dynamic_meshes.size()) continue;
            tc.pairs.push_back(ref);
        }
    }
    return tc;
}

struct LidarOrbitTestCfg {
    int   dynamic_mesh_index = -1;
    float orbit_radius       = 2.0f;
    float orbit_speed        = 1.0f;
    int   seed               = 0;
    struct LidarRef { int robot_idx = -1; int lidar_idx = -1; };
    std::vector<LidarRef> lidar_candidates;
};

static std::optional<LidarOrbitTestCfg> parseLidarOrbitTest(const json& j,
                                                              const RunConfig& cfg) {
    if (!j.contains("lidar_orbit_test")) return std::nullopt;
    const auto& lo = j["lidar_orbit_test"];
    LidarOrbitTestCfg tc;
    if (lo.contains("dynamic_mesh_index")) tc.dynamic_mesh_index = lo["dynamic_mesh_index"].get<int>();
    if (lo.contains("orbit_radius")) tc.orbit_radius = lo["orbit_radius"].get<float>();
    if (lo.contains("orbit_speed"))  tc.orbit_speed  = lo["orbit_speed"].get<float>();
    if (lo.contains("seed"))         tc.seed         = lo["seed"].get<int>();
    if (lo.contains("lidar_indices") && lo["lidar_indices"].is_array()) {
        for (const auto& arr : lo["lidar_indices"]) {
            if (!arr.is_array() || arr.size() != 2) continue;
            LidarOrbitTestCfg::LidarRef ref;
            ref.robot_idx = arr[0].get<int>();
            ref.lidar_idx = arr[1].get<int>();
            if (ref.robot_idx < 0 || ref.robot_idx >= (int)cfg.robots.size()) continue;
            if (ref.lidar_idx < 0 || ref.lidar_idx >= (int)cfg.robots[ref.robot_idx].lidars.size()) continue;
            tc.lidar_candidates.push_back(ref);
        }
    }
    return tc;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

static double elapsed_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// R = R_y(yaw) * R_x(pitch) * R_z(roll)
static void rotationFromEuler(float yaw, float pitch, float roll, Pose& p) {
    float cY = cosf(yaw),   sY = sinf(yaw);
    float cP = cosf(pitch), sP = sinf(pitch);
    float cR = cosf(roll),  sR = sinf(roll);
    p.right   = { cY*cR + sY*sP*sR,  cP*sR, -sY*cR + cY*sP*sR };
    p.up      = {-cY*sR + sY*sP*cR,  cP*cR,  sY*sR + cY*sP*cR };
    p.forward = { sY*cP,            -sP,      cY*cP             };
}

static void computeLidarWorldPose(const Pose& r, LidarCfg& lc) {
    const Pose& l = lc.l_pose;
    lc.w_pose.pos.x = r.pos.x + r.right.x*l.pos.x + r.up.x*l.pos.y + r.forward.x*l.pos.z;
    lc.w_pose.pos.y = r.pos.y + r.right.y*l.pos.x + r.up.y*l.pos.y + r.forward.y*l.pos.z;
    lc.w_pose.pos.z = r.pos.z + r.right.z*l.pos.x + r.up.z*l.pos.y + r.forward.z*l.pos.z;
    lc.w_pose.forward.x = r.right.x*l.forward.x + r.up.x*l.forward.y + r.forward.x*l.forward.z;
    lc.w_pose.forward.y = r.right.y*l.forward.x + r.up.y*l.forward.y + r.forward.y*l.forward.z;
    lc.w_pose.forward.z = r.right.z*l.forward.x + r.up.z*l.forward.y + r.forward.z*l.forward.z;
    lc.w_pose.up.x = r.right.x*l.up.x + r.up.x*l.up.y + r.forward.x*l.up.z;
    lc.w_pose.up.y = r.right.y*l.up.x + r.up.y*l.up.y + r.forward.y*l.up.z;
    lc.w_pose.up.z = r.right.z*l.up.x + r.up.z*l.up.y + r.forward.z*l.up.z;
    lc.w_pose.right.x = lc.w_pose.up.y*lc.w_pose.forward.z - lc.w_pose.up.z*lc.w_pose.forward.y;
    lc.w_pose.right.y = lc.w_pose.up.z*lc.w_pose.forward.x - lc.w_pose.up.x*lc.w_pose.forward.z;
    lc.w_pose.right.z = lc.w_pose.up.x*lc.w_pose.forward.y - lc.w_pose.up.y*lc.w_pose.forward.x;
}

// ---------------------------------------------------------------------------
// OBJ loader
// ---------------------------------------------------------------------------

// Load OBJ and extract per-triangle Kd diffuse colors from MTL materials.
// out_colors is filled with count*3 floats (RGB per triangle); falls back to grey if no MTL.
static std::vector<Tri3> loadOBJ(const std::string& path,
                                  std::vector<float>& out_colors) {
    tinyobj::ObjReaderConfig rcfg; rcfg.triangulate = true;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, rcfg)) {
        if (!reader.Error().empty()) throw std::runtime_error("tinyobjloader: " + reader.Error());
        throw std::runtime_error("Failed to parse: " + path);
    }
    if (!reader.Warning().empty()) fprintf(stderr, "[tinyobjloader] %s\n", reader.Warning().c_str());
    const auto& attrib    = reader.GetAttrib();
    const auto& shapes    = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    std::vector<Tri3> tris;
    out_colors.clear();
    for (const auto& shape : shapes) {
        size_t off = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            Tri3 tri;
            for (int k = 0; k < 3; ++k) {
                int vi = shape.mesh.indices[off + k].vertex_index * 3;
                tri.v[k][0] = attrib.vertices[vi];
                tri.v[k][1] = attrib.vertices[vi + 1];
                tri.v[k][2] = attrib.vertices[vi + 2];
            }
            off += 3;
            tris.push_back(tri);

            int mat_id = (f < shape.mesh.material_ids.size()) ? shape.mesh.material_ids[f] : -1;
            if (mat_id >= 0 && mat_id < (int)materials.size()) {
                out_colors.push_back(materials[mat_id].diffuse[0]);
                out_colors.push_back(materials[mat_id].diffuse[1]);
                out_colors.push_back(materials[mat_id].diffuse[2]);
            } else {
                out_colors.push_back(0.7f);
                out_colors.push_back(0.7f);
                out_colors.push_back(0.7f);
            }
        }
    }
    return tris;
}

// ---------------------------------------------------------------------------
// CSV writer
// ---------------------------------------------------------------------------

static void writeCSV(const std::string& path, const std::vector<GrcaHit>& hits,
                     const std::vector<int>& ro_ids, const std::vector<int>& li_ids) {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) throw std::runtime_error("Cannot open output file: " + path);
    fprintf(fp, "ro_id,li_id,dx,dy,dz,dist,hx,hy,hz\n");
    for (size_t i = 0; i < hits.size(); ++i)
        fprintf(fp, "%d,%d,%g,%g,%g,%g,%g,%g,%g\n",
                ro_ids[i], li_ids[i],
                hits[i].dx, hits[i].dy, hits[i].dz,
                hits[i].dist, hits[i].hx, hits[i].hy, hits[i].hz);
    fclose(fp);
}

// ---------------------------------------------------------------------------
// Timing report
// ---------------------------------------------------------------------------

static void printTimings(int runIdx, int batchIdx,
                        const RunConfig& cfg, int totalRays,
                        size_t staticTris, size_t dynTris, const FrameTimings& t,
                        const std::string& sessionLabel) {
    const std::string& backend = cfg.backend;
    bool isCpuBackend = (backend == "grca-cpu" || backend == "embree" || backend == "tinybvh-cpu"
                         || backend == "hybrid-cpu");
    double traceMs  = isCpuBackend ? t.ms_trace_cpu : t.ms_trace_gpu;
    const char* traceTag = isCpuBackend ? "cpu" : "gpu";
    fprintf(stderr, "\n--- Breakdown [Run %d, Batch %d, %s] (%d rays, %zu static + %zu dyn tris) ---\n",
            runIdx, batchIdx, sessionLabel.c_str(), totalRays, staticTris, dynTris);
    if (t.ms_dynamic_upload  > 0) fprintf(stderr, "  Dynamic upload:  %8.2f ms  (cpu)\n",  t.ms_dynamic_upload);
    if (t.ms_bvh_rebuild_gpu > 0) fprintf(stderr, "  BVH Rebuild:     %8.2f ms  (gpu)\n",  t.ms_bvh_rebuild_gpu);
    if (t.ms_bvh_refit_gpu   > 0) fprintf(stderr, "  BVH Refit:       %8.2f ms  (gpu)\n",  t.ms_bvh_refit_gpu);
    if (t.ms_bvh_rebuild_cpu > 0) fprintf(stderr, "  BVH Rebuild:     %8.2f ms  (cpu)\n",  t.ms_bvh_rebuild_cpu);
    if (t.ms_bvh_refit_cpu   > 0) fprintf(stderr, "  BVH Refit:       %8.2f ms  (cpu)\n",  t.ms_bvh_refit_cpu);
    if (backend != "grca-cpu")     fprintf(stderr, "  Trace:           %8.2f ms  (%s)\n",   traceMs, traceTag);
    if (t.ms_sync_readback > 0) fprintf(stderr, "  CPU-GPU sync:    %8.2f ms  (cpu)\n",    t.ms_sync_readback);
    if (t.ms_readback_cpu  > 0) fprintf(stderr, "  Readback:        %8.2f ms  (cpu, %.2f MB)\n", t.ms_readback_cpu, t.mb_readback);
    if (t.ms_fillhits      > 0) fprintf(stderr, "  Hit fill:        %8.2f ms  (cpu)\n",    t.ms_fillhits);
    fprintf(stderr, "  ----------------------------------------\n");
    fprintf(stderr, "  Lidar sim:       %8.2f ms  (%s)\n", t.ms_lidar_sim_cpu, backendLabel(cfg).c_str());
    fprintf(stderr, "  Total wall-clock:%8.2f ms  (incl. mesh prep)\n", t.ms_total_cpu);
    fprintf(stderr, "  Sim frame rate:  %8.2f fps\n", t.sim_frame_rate);
    fprintf(stderr, "  Real-time factor:%8.3f\n\n", t.real_time_factor);
}

// ---------------------------------------------------------------------------
// Lidar / config parsers
// ---------------------------------------------------------------------------

static LidarCfg parseLidar(const json& jl) {
    LidarCfg lc;
    if (jl.contains("hmin"))      lc.hmin      = jl["hmin"].get<float>();
    if (jl.contains("hmax"))      lc.hmax      = jl["hmax"].get<float>();
    if (jl.contains("hnum"))      lc.hnum      = jl["hnum"].get<int>();
    if (jl.contains("vmin"))      lc.vmin      = jl["vmin"].get<float>();
    if (jl.contains("vmax"))      lc.vmax      = jl["vmax"].get<float>();
    if (jl.contains("vnum"))      lc.vnum      = jl["vnum"].get<int>();
    if (jl.contains("range_min")) lc.range_min = jl["range_min"].get<float>();
    if (jl.contains("range_max")) lc.range_max = jl["range_max"].get<float>();
    if (jl.contains("position") && jl["position"].is_array()) {
        lc.l_pose.pos = {jl["position"][0].get<float>(), jl["position"][1].get<float>(), jl["position"][2].get<float>()};
    }
    if (jl.contains("forward") && jl["forward"].is_array()) {
        lc.l_pose.forward = {jl["forward"][0].get<float>(), jl["forward"][1].get<float>(), jl["forward"][2].get<float>()};
    }
    if (jl.contains("up") && jl["up"].is_array()) {
        lc.l_pose.up = {jl["up"][0].get<float>(), jl["up"][1].get<float>(), jl["up"][2].get<float>()};
    }
    // right = up × forward
    lc.l_pose.right = {
        lc.l_pose.up.y*lc.l_pose.forward.z - lc.l_pose.up.z*lc.l_pose.forward.y,
        lc.l_pose.up.z*lc.l_pose.forward.x - lc.l_pose.up.x*lc.l_pose.forward.z,
        lc.l_pose.up.x*lc.l_pose.forward.y - lc.l_pose.up.y*lc.l_pose.forward.x
    };
    return lc;
}

// Parse a scene or flat config JSON into RunConfig.
// lidar_sim_rate and auto_hit_view_record_frame are set as outputs when found.
static RunConfig parseConfig(const json& j, float& lidar_sim_rate, int& auto_hit_view_record_frame) {
    RunConfig cfg;

    // random_dynamic_test (may live in scene or flat config)
    if (j.contains("random_dynamic_test")) {
        const auto& rdt = j["random_dynamic_test"];
        cfg.seed = rdt.value("seed", 0);
        cfg.random_dynamic_from_static_aabb = true;
        if (rdt.contains("pos_interpolate_frames"))
            cfg.random_dynamic_pos_interpolate_frames   = rdt["pos_interpolate_frames"].get<int>();
        if (rdt.contains("rot_interpolate_frames"))
            cfg.random_dynamic_rot_interpolate_frames   = rdt["rot_interpolate_frames"].get<int>();
        if (rdt.contains("scale_interpolate_frames"))
            cfg.random_dynamic_scale_interpolate_frames = rdt["scale_interpolate_frames"].get<int>();
        if (rdt.contains("scale_min")) cfg.random_dynamic_scale_min = rdt["scale_min"].get<float>();
        if (rdt.contains("scale_max")) cfg.random_dynamic_scale_max = rdt["scale_max"].get<float>();
        if (rdt.contains("tri_chaos")) {
            auto& v = rdt["tri_chaos"];
            cfg.random_tri_chaos = v.is_boolean() ? (v.get<bool>() ? 1 : 0) : v.get<int>();
        }
        if (rdt.contains("robots_share_pose")) cfg.random_robots_share_pose = rdt["robots_share_pose"].get<bool>();
    }

    if (j.contains("output"))  cfg.output  = j["output"].get<std::string>();
    if (j.contains("backend")) cfg.backend = j["backend"].get<std::string>();

    // cuda / optix backend options (present in flat config)
    if (j.contains("cuda_debug"))
        cfg.debug = j["cuda_debug"].get<bool>();
    if (j.contains("cuda")) {
        const auto& cj = j["cuda"];
        if (cj.contains("debug"))             cfg.debug                  = cj["debug"].get<bool>();
        if (cj.contains("sat_max_sweep_diff"))   cfg.cuda_sat_max_sweep_diff   = cj["sat_max_sweep_diff"].get<int>();
        if (cj.contains("sat_max_channel_diff")) cfg.cuda_sat_max_channel_diff = cj["sat_max_channel_diff"].get<int>();
    }
    if (j.contains("cpu")) {
        const auto& gj = j["cpu"];
        if (gj.contains("sat_max_sweep_diff"))   cfg.cpu_sat_max_sweep_diff    = gj["sat_max_sweep_diff"].get<int>();
        if (gj.contains("sat_max_channel_diff")) cfg.cpu_sat_max_channel_diff  = gj["sat_max_channel_diff"].get<int>();
    }
    if (j.contains("optix")) {
        const auto& oj = j["optix"];
        if (oj.contains("bvh_rebuild_after_frames")) cfg.optix_rebuild_after_frames = oj["bvh_rebuild_after_frames"].get<int>();
        if (oj.contains("bvh_compaction_on"))        cfg.optix_compaction           = oj["bvh_compaction_on"].get<bool>();
    }
    if (j.contains("embree")) {
        const auto& ej = j["embree"];
        if (ej.contains("bvh_rebuild_after_frames"))  cfg.embree_rebuild_after_frames = ej["bvh_rebuild_after_frames"].get<int>();
        if (ej.contains("bvh_compaction_on"))         cfg.embree_compaction = ej["bvh_compaction_on"].get<bool>();
    }
    if (j.contains("tinybvh")) {
        const auto& tj = j["tinybvh"];
        if (tj.contains("bvh_rebuild_after_frames")) {
            cfg.tinybvh_cpu_rebuild_after_frames = tj["bvh_rebuild_after_frames"].get<int>();
            cfg.tinybvh_gpu_rebuild_after_frames = tj["bvh_rebuild_after_frames"].get<int>();
        }
        if (tj.contains("cpu_rebuild_after_frames")) cfg.tinybvh_cpu_rebuild_after_frames = tj["cpu_rebuild_after_frames"].get<int>();
        if (tj.contains("gpu_rebuild_after_frames")) cfg.tinybvh_gpu_rebuild_after_frames = tj["gpu_rebuild_after_frames"].get<int>();
    }

    // robots
    if (j.contains("robots")) {
        for (const auto& jr : j["robots"]) {
            RobotCfg robot;
            if (jr.contains("w_position") && jr["w_position"].is_array()) {
                robot.w_pose.pos = {jr["w_position"][0].get<float>(), jr["w_position"][1].get<float>(), jr["w_position"][2].get<float>()};
            }
            if (jr.contains("w_forward") && jr["w_forward"].is_array()) {
                robot.w_pose.forward = {jr["w_forward"][0].get<float>(), jr["w_forward"][1].get<float>(), jr["w_forward"][2].get<float>()};
            }
            if (jr.contains("w_up") && jr["w_up"].is_array()) {
                robot.w_pose.up = {jr["w_up"][0].get<float>(), jr["w_up"][1].get<float>(), jr["w_up"][2].get<float>()};
            }
            // right = up × forward
            robot.w_pose.right = {
                robot.w_pose.up.y*robot.w_pose.forward.z - robot.w_pose.up.z*robot.w_pose.forward.y,
                robot.w_pose.up.z*robot.w_pose.forward.x - robot.w_pose.up.x*robot.w_pose.forward.z,
                robot.w_pose.up.x*robot.w_pose.forward.y - robot.w_pose.up.y*robot.w_pose.forward.x
            };
            if (jr.contains("lidars"))
                for (const auto& jl : jr["lidars"]) robot.lidars.push_back(parseLidar(jl));
            cfg.robots.push_back(robot);
        }
    }

    // static meshes
    if (j.contains("static_meshes")) {
        for (const auto& jm : j["static_meshes"]) {
            StaticMesh sm;
            sm.file  = jm["file"].get<std::string>();
            sm.scale = jm.value("scale", 1.0f);
            if (jm.contains("w_position") && jm["w_position"].is_array())
                sm.w_pose.pos = {jm["w_position"][0].get<float>(), jm["w_position"][1].get<float>(), jm["w_position"][2].get<float>()};
            if (jm.contains("color") && jm["color"].is_array() && jm["color"].size() == 3)
                for (int i = 0; i < 3; ++i) sm.color[i] = jm["color"][i].get<float>();
            else { sm.color[0] = 0.7f; sm.color[1] = 0.7f; sm.color[2] = 0.7f; }
            sm.color_override = jm.value("color_override", false);
            cfg.static_meshes.push_back(sm);
        }
    }

    // dynamic meshes
    if (j.contains("dynamic_meshes")) {
        for (const auto& jm : j["dynamic_meshes"]) {
            DynamicMesh dm;
            dm.file  = jm["file"].get<std::string>();
            dm.scale = jm.value("scale", 1.0f);
            if (jm.contains("w_position") && jm["w_position"].is_array())
                dm.w_pose.pos = {jm["w_position"][0].get<float>(), jm["w_position"][1].get<float>(), jm["w_position"][2].get<float>()};
            if (jm.contains("w_forward") && jm["w_forward"].is_array())
                dm.w_pose.forward = {jm["w_forward"][0].get<float>(), jm["w_forward"][1].get<float>(), jm["w_forward"][2].get<float>()};
            if (jm.contains("w_up") && jm["w_up"].is_array())
                dm.w_pose.up = {jm["w_up"][0].get<float>(), jm["w_up"][1].get<float>(), jm["w_up"][2].get<float>()};
            dm.w_pose.right = {
                dm.w_pose.up.y*dm.w_pose.forward.z - dm.w_pose.up.z*dm.w_pose.forward.y,
                dm.w_pose.up.z*dm.w_pose.forward.x - dm.w_pose.up.x*dm.w_pose.forward.z,
                dm.w_pose.up.x*dm.w_pose.forward.y - dm.w_pose.up.y*dm.w_pose.forward.x
            };
            if (jm.contains("velocity") && jm["velocity"].is_array())
                for (int i = 0; i < 3; ++i) dm.velocity[i] = jm["velocity"][i].get<float>();
            if (jm.contains("color") && jm["color"].is_array() && jm["color"].size() == 3)
                for (int i = 0; i < 3; ++i) dm.color[i] = jm["color"][i].get<float>();
            else { dm.color[0] = 0.7f; dm.color[1] = 0.7f; dm.color[2] = 0.7f; }
            dm.color_override = jm.value("color_override", false);
            cfg.dynamic_meshes.push_back(dm);
        }
    }

    lidar_sim_rate = 10.0f;
    if (j.contains("lidar_sim_rate")) lidar_sim_rate = j["lidar_sim_rate"].get<float>();
    auto_hit_view_record_frame = -1;
    if (j.contains("auto_hit_view_record_frame")) auto_hit_view_record_frame = j["auto_hit_view_record_frame"].get<int>();
    if (j.contains("visualizer_on")) {
        auto& v = j["visualizer_on"];
        cfg.visualizer_on = v.is_boolean() ? v.get<bool>() : (v.get<int>() != 0);
    }
    if (j.contains("mesh_vis_on")) {
        auto& v = j["mesh_vis_on"];
        cfg.mesh_vis_on = v.is_boolean() ? (v.get<bool>() ? 1 : 0) : v.get<int>();
    }
    if (j.contains("cone_vis_on"))
        cfg.cone_vis_on = j["cone_vis_on"].get<bool>();
    return cfg;
}

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv)
{
#ifdef HAVE_CUDA
    size_t vram_main_start = 0;
    {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess)
            vram_main_start = t - f;
    }
#endif

    // -----------------------------------------------------------------------
    // 1. Locate and load config
    // -----------------------------------------------------------------------
    std::string configPath = "benchmark_config.json";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.size() >= 5 && a.substr(a.size() - 5) == ".json") { configPath = a; break; }
    }

    json j;
    try {
        std::ifstream f(configPath);
        if (!f.is_open()) throw std::runtime_error("Cannot open config file: " + configPath);
        f >> j;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n"
                  << "Usage: grca [benchmark_config.json] [--backend grca-cpu|grca-cuda|optix|optix-crti|embree|tinybvh-cpu|tinybvh-gpu|hybrid-cpu|hybrid-gpu|all|all_t|all_h|cpu|cpu_t|cpu_h|gpu|gpu_t|gpu_h|gpu-all|bvh|bvh_t|grca|hybrid]\n"
                  << "           [--out <prefix>] [--debug] [--refit] [--rebuild]\n"
                  << "           [--active tests[N] | tests[N]/batches[M]]\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 2. Detect config format
    // -----------------------------------------------------------------------
    // Derive config directory so relative file paths in config resolve correctly
    std::string configDir = ".";
    {
        auto sep = configPath.find_last_of("/\\");
        if (sep != std::string::npos) configDir = configPath.substr(0, sep);
    }

    auto loadJson = [&](const std::string& relPath) -> json {
        std::string full = configDir + "/" + relPath;
        std::ifstream f(full);
        if (!f.is_open()) throw std::runtime_error("Cannot open file: " + full);
        json out; f >> out; return out;
    };

    bool is_new_format = j.contains("active_tests") && j.contains("benchmark_robots");
    std::vector<std::string> benchmark_scenes_list, benchmark_robots_list, benchmark_dyn_meshes_list;
    std::vector<json> tests_list;
    if (is_new_format) {
        if (j.contains("benchmark_scenes"))
            for (const auto& s : j["benchmark_scenes"])
                benchmark_scenes_list.push_back(s.get<std::string>());
        for (const auto& r : j["benchmark_robots"])
            benchmark_robots_list.push_back(r.get<std::string>());
        if (j.contains("benchmark_dynamic_meshes"))
            for (const auto& d : j["benchmark_dynamic_meshes"])
                benchmark_dyn_meshes_list.push_back(d.get<std::string>());
        for (const auto& tf : j["active_tests"])
            tests_list.push_back(loadJson(tf.get<std::string>()));
    } else {
        tests_list.push_back(j);
    }

    // -----------------------------------------------------------------------
    // 4. Parse CLI overrides (stored, applied per-run)
    // -----------------------------------------------------------------------
    std::string cli_backend, cli_out, cli_active;
    bool cli_debug = false, cli_refit = false, cli_rebuild = false;
    for (int i = 1; i < argc; ++i) {
        if      (!std::strcmp(argv[i], "--backend") && i+1 < argc) cli_backend = argv[++i];
        else if (!std::strcmp(argv[i], "--out")     && i+1 < argc) cli_out     = argv[++i];
        else if (!std::strcmp(argv[i], "--active")  && i+1 < argc) cli_active  = argv[++i];
        else if (!std::strcmp(argv[i], "--debug"))   cli_debug   = true;
        else if (!std::strcmp(argv[i], "--refit"))   cli_refit   = true;
        else if (!std::strcmp(argv[i], "--rebuild")) cli_rebuild = true;
    }

    // -----------------------------------------------------------------------
    // 8. Build active filter (constant for all benchmark pairs)
    // -----------------------------------------------------------------------
    auto parse_selector = [](const std::string& s) -> std::pair<int,int> {
        auto extract = [&](const std::string& key) -> int {
            auto p = s.find(key + "[");
            if (p == std::string::npos) return -1;
            auto q = s.find(']', p);
            if (q == std::string::npos) return -1;
            try { return std::stoi(s.substr(p + key.size() + 1, q - p - key.size() - 1)); }
            catch (...) { return -1; }
        };
        return {extract("tests"), extract("batches")};
    };

    std::vector<std::pair<int,int>> active_filters;
    if (!cli_active.empty()) {
        active_filters.push_back(parse_selector(cli_active));
    } else if (!is_new_format && j.contains("active_tests") && j["active_tests"].is_array()) {
        for (const auto& sel : j["active_tests"])
            active_filters.push_back(parse_selector(sel.get<std::string>()));
    }

    if (!active_filters.empty()) {
        fprintf(stderr, "Active filters:\n");
        for (const auto& [ti, bi] : active_filters) {
            std::string ts = ti < 0 ? "*" : std::to_string(ti);
            std::string bs = bi < 0 ? "*" : std::to_string(bi);
            fprintf(stderr, "  tests[%s]/batches[%s]\n", ts.c_str(), bs.c_str());
        }
    }

    // Benchmark-level tag and archive (start time of the full run, shared by all sessions)
    std::string bench_tag, bench_archive;
    {
        auto bench_tp = std::chrono::system_clock::now();
        std::time_t bench_t = std::chrono::system_clock::to_time_t(bench_tp);
        std::tm bench_tm{};
        localtime_r(&bench_t, &bench_tm);
        std::ostringstream ts;
        ts << std::put_time(&bench_tm, "%Y%m%d_%H%M%S")
           << "-" << sanitize_path_component(get_username())
           << "-" << sanitize_path_component(get_hostname());
        bench_tag     = ts.str();
        bench_archive = "benchmark/" + bench_tag + ".zip";
    }
    std::vector<std::string> session_zips;
#ifdef HAVE_CUDA
    // batch label → max VRAM (total-free) across all frames
    std::vector<std::pair<std::string, size_t>> all_batch_max_vram;
#endif

    // -----------------------------------------------------------------------
    // Outer loop: one full session per (scene, dyn_mesh, robot) combination
    // -----------------------------------------------------------------------
    const bool no_benchmark_scenes = is_new_format && benchmark_scenes_list.empty();

    VisState* vis = nullptr;  // lives across all sessions, destroyed after all loops
    size_t scene_idx = 0;
    for (const auto& cur_scene_path : ((is_new_format && !benchmark_scenes_list.empty()) ? benchmark_scenes_list : std::vector<std::string>{""})) {
    size_t dyn_idx = 0;
    for (const auto& cur_dyn_mesh_path : (is_new_format && !benchmark_dyn_meshes_list.empty() ? benchmark_dyn_meshes_list : std::vector<std::string>{""})) {
    size_t robot_idx = 0;
    for (const auto& cur_robot_path : (is_new_format ? benchmark_robots_list : std::vector<std::string>{""})) {

    // -----------------------------------------------------------------------
    // 3. Parse scene config for this (scene, robot) pair
    // -----------------------------------------------------------------------
    json scene_j;
    if (is_new_format) {
        scene_j = cur_scene_path.empty() ? json::object() : loadJson(cur_scene_path);
        scene_j["robots"] = loadJson(cur_robot_path);
        if (!cur_dyn_mesh_path.empty())
            scene_j["dynamic_meshes"] = loadJson(cur_dyn_mesh_path);
    } else {
        scene_j = j;
    }

    float lidar_sim_rate_global = 10.0f;
    int   auto_hit_view_global  = -1;
    RunConfig scene_cfg = parseConfig(scene_j, lidar_sim_rate_global, auto_hit_view_global);

    // Top-level visualizer and camera settings (from benchmark_config.json)
    bool  vis_on_global    = false;
    int   mesh_vis_global  = 0;
    bool  cone_vis_global  = false;
    bool  has_cam_pos = false, has_cam_fwd = false;
    float cam_pos_global[3] = {}, cam_fwd_global[3] = {0,0,-1}, cam_up_global[3] = {0,1,0};
    {
        auto read_vis = [&](const json& jj) {
            if (jj.contains("visualizer_on")) {
                auto& v = jj["visualizer_on"];
                vis_on_global = v.is_boolean() ? v.get<bool>() : (v.get<int>() != 0);
            }
            if (jj.contains("mesh_vis_on")) {
                auto& v = jj["mesh_vis_on"];
                mesh_vis_global = v.is_boolean() ? (v.get<bool>() ? 1 : 0) : v.get<int>();
            }
            if (jj.contains("cone_vis_on"))
                cone_vis_global = jj["cone_vis_on"].get<bool>();
            if (jj.contains("w_cam_position") && jj["w_cam_position"].is_array()) {
                cam_pos_global[0] = jj["w_cam_position"][0].get<float>();
                cam_pos_global[1] = jj["w_cam_position"][1].get<float>();
                cam_pos_global[2] = jj["w_cam_position"][2].get<float>();
                has_cam_pos = true;
            }
            if (jj.contains("w_cam_forward") && jj["w_cam_forward"].is_array()) {
                cam_fwd_global[0] = jj["w_cam_forward"][0].get<float>();
                cam_fwd_global[1] = jj["w_cam_forward"][1].get<float>();
                cam_fwd_global[2] = jj["w_cam_forward"][2].get<float>();
                has_cam_fwd = true;
            }
            if (jj.contains("w_cam_up") && jj["w_cam_up"].is_array()) {
                cam_up_global[0] = jj["w_cam_up"][0].get<float>();
                cam_up_global[1] = jj["w_cam_up"][1].get<float>();
                cam_up_global[2] = jj["w_cam_up"][2].get<float>();
            }
        };
        read_vis(j);                     // benchmark_config.json
        if (!tests_list.empty()) read_vis(tests_list[0]); // first test can still override
    }

    // Use first tests entry for top-level defaults
    const json& first_run_j = tests_list[0];
    if (first_run_j.contains("lidar_sim_rate"))
        lidar_sim_rate_global = first_run_j["lidar_sim_rate"].get<float>();
    if (first_run_j.contains("auto_hit_view_record_frame"))
        auto_hit_view_global = first_run_j["auto_hit_view_record_frame"].get<int>();

    if (scene_cfg.robots.empty()) {
        std::cerr << "Error: no robots defined in config.\n";
        return 1;
    }

    // -----------------------------------------------------------------------
    // 5. Load meshes once (shared across all batches)
    // -----------------------------------------------------------------------
    std::vector<Tri3> static_tris;
    std::vector<std::vector<Tri3>> static_vis_tris;
    std::vector<std::vector<float>> static_vis_colors;  // per-triangle RGB from MTL
    std::vector<std::vector<Tri3>> dyn_mesh_tris;
    std::vector<std::vector<float>> dyn_mesh_colors;    // per-triangle RGB from MTL

    for (const auto& sm : scene_cfg.static_meshes) {
        std::vector<float> tri_colors;
        auto mesh = loadOBJ(sm.file, tri_colors);
        for (auto& tri : mesh)
            for (int k = 0; k < 3; ++k) {
                tri.v[k][0] = tri.v[k][0] * sm.scale + sm.w_pose.pos.x;
                tri.v[k][1] = tri.v[k][1] * sm.scale + sm.w_pose.pos.y;
                tri.v[k][2] = tri.v[k][2] * sm.scale + sm.w_pose.pos.z;
            }
        static_vis_tris.push_back(mesh);
        static_vis_colors.push_back(std::move(tri_colors));
        static_tris.insert(static_tris.end(), mesh.begin(), mesh.end());
    }
    for (const auto& dm : scene_cfg.dynamic_meshes) {
        std::vector<float> tri_colors;
        dyn_mesh_tris.push_back(loadOBJ(dm.file, tri_colors));
        dyn_mesh_colors.push_back(std::move(tri_colors));
    }

    // Build initial combined triangle soup (for GPU backend sizing/BVH init)
    std::vector<Tri3> init_tris = static_tris;
    for (size_t i = 0; i < scene_cfg.dynamic_meshes.size(); ++i) {
        const auto& dm = scene_cfg.dynamic_meshes[i];
        std::vector<Tri3> dyn = dyn_mesh_tris[i];
        for (auto& tri : dyn)
            for (int k = 0; k < 3; ++k) {
                float vx = tri.v[k][0] * dm.scale;
                float vy = tri.v[k][1] * dm.scale;
                float vz = tri.v[k][2] * dm.scale;
                tri.v[k][0] = dm.w_pose.right.x*vx + dm.w_pose.up.x*vy + dm.w_pose.forward.x*vz + dm.w_pose.pos.x;
                tri.v[k][1] = dm.w_pose.right.y*vx + dm.w_pose.up.y*vy + dm.w_pose.forward.y*vz + dm.w_pose.pos.y;
                tri.v[k][2] = dm.w_pose.right.z*vx + dm.w_pose.up.z*vy + dm.w_pose.forward.z*vz + dm.w_pose.pos.z;
            }
        init_tris.insert(init_tris.end(), dyn.begin(), dyn.end());
    }

    // Per-dynamic-mesh local-space AABBs (tri_chaos mode 2) and precomputed centroids
    std::vector<std::array<float,6>> dyn_mesh_aabbs;
    std::vector<std::vector<std::array<float,3>>> dyn_mesh_centroids;
    for (size_t i = 0; i < scene_cfg.dynamic_meshes.size(); ++i) {
        float xmn = std::numeric_limits<float>::max(),    ymn = xmn, zmn = xmn;
        float xmx = std::numeric_limits<float>::lowest(), ymx = xmx, zmx = xmx;
        const size_t ntris = dyn_mesh_tris[i].size();
        std::vector<std::array<float,3>> cents(ntris);
        for (size_t ji = 0; ji < ntris; ++ji) {
            const auto& tri = dyn_mesh_tris[i][ji];
            for (int k = 0; k < 3; ++k) {
                xmn=std::min(xmn,tri.v[k][0]); xmx=std::max(xmx,tri.v[k][0]);
                ymn=std::min(ymn,tri.v[k][1]); ymx=std::max(ymx,tri.v[k][1]);
                zmn=std::min(zmn,tri.v[k][2]); zmx=std::max(zmx,tri.v[k][2]);
            }
            cents[ji] = { (tri.v[0][0]+tri.v[1][0]+tri.v[2][0])*(1.f/3.f),
                           (tri.v[0][1]+tri.v[1][1]+tri.v[2][1])*(1.f/3.f),
                           (tri.v[0][2]+tri.v[1][2]+tri.v[2][2])*(1.f/3.f) };
        }
        dyn_mesh_aabbs.push_back({xmn,ymn,zmn,xmx,ymx,zmx});
        dyn_mesh_centroids.push_back(std::move(cents));
    }

    // Pre-compute static AABB (reused by any batch with random_dynamic_from_static_aabb)
    float static_aabb[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    bool  has_static_aabb = false;
    if (!static_tris.empty()) {
        float xmn = std::numeric_limits<float>::max(),    ymn = xmn, zmn = xmn;
        float xmx = std::numeric_limits<float>::lowest(), ymx = xmx, zmx = xmx;
        for (const auto& tri : static_tris)
            for (int k = 0; k < 3; ++k) {
                xmn = std::min(xmn, tri.v[k][0]); xmx = std::max(xmx, tri.v[k][0]);
                ymn = std::min(ymn, tri.v[k][1]); ymx = std::max(ymx, tri.v[k][1]);
                zmn = std::min(zmn, tri.v[k][2]); zmx = std::max(zmx, tri.v[k][2]);
            }
        static_aabb[0]=xmn; static_aabb[1]=ymn; static_aabb[2]=zmn;
        static_aabb[3]=xmx; static_aabb[4]=ymx; static_aabb[5]=zmx;
        has_static_aabb = true;
    } else if (no_benchmark_scenes) {
        // No benchmark scene provided: fall back to a fixed 10x10x10 world box.
        static_aabb[0] = -5.f; static_aabb[1] = -5.f; static_aabb[2] = -5.f;
        static_aabb[3] =  5.f; static_aabb[4] =  5.f; static_aabb[5] =  5.f;
        has_static_aabb = true;
    }
    const bool show_fallback_static_aabb = no_benchmark_scenes && has_static_aabb && scene_cfg.static_meshes.empty();

    // -----------------------------------------------------------------------
    // 6. Create session directory
    // -----------------------------------------------------------------------
    auto path_stem = [](const std::string& p) {
        auto s = p;
        auto slash = s.find_last_of('/');
        if (slash != std::string::npos) s = s.substr(slash + 1);
        auto dot = s.find_last_of('.');
        if (dot != std::string::npos) s = s.substr(0, dot);
        return s;
    };
    std::string session_label = "[scene"  + std::to_string(scene_idx)  + "]" + path_stem(cur_scene_path)
                              + "-[dyn"   + std::to_string(dyn_idx)    + "]" + path_stem(cur_dyn_mesh_path)
                              + "-[robot" + std::to_string(robot_idx)  + "]" + path_stem(cur_robot_path);
    std::string session_dir = "benchmark/" + bench_tag + "-" + session_label;
    ensure_dir(session_dir);
    fprintf(stderr, "Session: %s\n", session_dir.c_str());
    {
        int sess_rays = 0;
        for (const auto& robot : scene_cfg.robots)
            for (const auto& lidar : robot.lidars)
                sess_rays += lidar.hnum * lidar.vnum;
        size_t dyn_tri_count = 0;
        for (const auto& t : dyn_mesh_tris) dyn_tri_count += t.size();
        write_session_meta_csv(session_dir + "/session_meta.csv", session_dir,
                               configPath, j, is_new_format,
                               cur_scene_path, cur_robot_path, cur_dyn_mesh_path,
                               scene_cfg, static_tris.size(), dyn_tri_count, sess_rays);
        // Snapshot: copy only the files actually used, preserving relative paths.
        copy_file_to_session(configPath, session_dir);
        if (is_new_format) {
            if (!cur_scene_path.empty())
                copy_config_file(configDir + "/" + cur_scene_path, cur_scene_path, session_dir);
            copy_config_file(configDir + "/" + cur_robot_path, cur_robot_path, session_dir);
            if (!cur_dyn_mesh_path.empty())
                copy_config_file(configDir + "/" + cur_dyn_mesh_path, cur_dyn_mesh_path, session_dir);
            for (const auto& tf : j["active_tests"]) {
                std::string rp = tf.get<std::string>();
                copy_config_file(configDir + "/" + rp, rp, session_dir);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. Backend state pointers (declared once, reset between runs)
    // -----------------------------------------------------------------------
    CpuState* cpuState = nullptr;
#ifdef HAVE_CUDA
    CudaState* cudaState = nullptr;
#endif
#ifdef HAVE_OPTIX
    OptixState* optixState = nullptr;
#endif
#ifdef HAVE_EMBREE
    EmbreeState* embreeState = nullptr;
#endif
#ifdef HAVE_TINYBVH
    TinyBVHCPUState* tinybvhCpuState = nullptr;
    TinyBVHGPUState* tinybvhGpuState = nullptr;
#endif
#ifdef HAVE_EMBREE
    HybridCpuState* hybridCpuState = nullptr;
#endif
#if defined(HAVE_OPTIX) && defined(HAVE_CUDA)
    HybridGpuState* hybridGpuState = nullptr;
#endif

    // -----------------------------------------------------------------------
    // 9. tests loop  (each entry is an independent run with its own
    //    recording settings, backend options, and batches list)
    // -----------------------------------------------------------------------
    bool skip_plot = j.value("skip_plot", true);
    int  json_bat_capacity = j.value("bat_capacity", 800000);
    bool multi_run = tests_list.size() > 1;
    for (size_t runIdx = 0; runIdx < tests_list.size(); ++runIdx) {
        if (!active_filters.empty()) {
            bool match = false;
            for (const auto& [ti, bi] : active_filters)
                if (ti < 0 || (int)runIdx == ti) { match = true; break; }
            if (!match) continue;
        }
        const json& cur_run_j = tests_list[runIdx];

        // Base output directory for this run
        std::string run_base = session_dir + "/run_" + std::to_string(runIdx);
        ensure_dir(run_base);
        if (multi_run)
            fprintf(stderr, "\n=== Run %zu / %zu ===\n", runIdx + 1, tests_list.size());

        // Extract recording settings for this run
        int record_frames_val = -1;
        if (cur_run_j.contains("record_frames"))
            record_frames_val = cur_run_j["record_frames"].get<int>();
        int perf_record_frames_global     = record_frames_val;
        int hit_dist_record_frames_global = record_frames_val;

        bool skip_validation = j.value("skip_validation", true);
        if (skip_validation)
            hit_dist_record_frames_global = -1;

        // Build the set of frames to sample for hit-dist (fixed per run, shared across batches)
        std::unordered_set<int> hit_dist_sample_frames;
        if (hit_dist_record_frames_global > 0) {
            int n_samples = std::max(1, hit_dist_record_frames_global / 10);
            std::mt19937 rng(0xd157);
            std::uniform_int_distribution<int> dist(0, hit_dist_record_frames_global - 1);
            while ((int)hit_dist_sample_frames.size() < n_samples)
                hit_dist_sample_frames.insert(dist(rng));
        }

        std::optional<PoseRecordFramesCfg> pose_record_global;
        int pose_record_frames_count_global = -1;
        if (!skip_validation) {
            pose_record_global = parsePoseRecordFrames(cur_run_j, scene_cfg);
            if (pose_record_global->frames > -1)
                pose_record_frames_count_global = pose_record_global->frames;
        }

        // Build batch list for this run
        std::vector<json> batches;
        if (cur_run_j.contains("batches") && cur_run_j["batches"].is_array())
            for (const auto& b : cur_run_j["batches"]) batches.push_back(b);
        else
            batches.push_back(cur_run_j);

        // -----------------------------------------------------------------------
        // Visualizer: initialize once across all sessions, update meshes per scene
        // -----------------------------------------------------------------------
        bool run_vis_on = vis_on_global || scene_cfg.visualizer_on;
        int run_mesh_vis = mesh_vis_global ? mesh_vis_global : scene_cfg.mesh_vis_on;
        if (cur_run_j.contains("visualizer_on")) {
            auto& v = cur_run_j["visualizer_on"];
            run_vis_on = v.is_boolean() ? v.get<bool>() : (v.get<int>() != 0);
        }
        if (cur_run_j.contains("mesh_vis_on")) {
            auto& v = cur_run_j["mesh_vis_on"];
            run_mesh_vis = v.is_boolean() ? (v.get<bool>() ? 1 : 0) : v.get<int>();
        }
        bool run_cone_vis = cone_vis_global || scene_cfg.cone_vis_on;
        if (cur_run_j.contains("cone_vis_on"))
            run_cone_vis = cur_run_j["cone_vis_on"].get<bool>();
        std::vector<VisAABB> vis_aabbs;
        if (run_vis_on) {
            scene_cfg.visualizer_on = true;
            scene_cfg.mesh_vis_on   = run_mesh_vis;
            scene_cfg.cone_vis_on   = run_cone_vis;
            if (!vis)
                vis = vis_init(scene_cfg);
            if (has_cam_pos || has_cam_fwd)
                vis_set_camera(vis, cam_pos_global,
                               has_cam_fwd ? cam_fwd_global : nullptr,
                               cam_up_global);
            std::vector<VisMesh> svm;
            for (size_t i = 0; i < scene_cfg.static_meshes.size(); ++i) {
                const auto& sm = scene_cfg.static_meshes[i];
                VisMesh vm;
                vm.tris       = static_vis_tris[i].data();
                vm.count      = static_vis_tris[i].size();
                vm.color[0]   = sm.color[0]; vm.color[1] = sm.color[1]; vm.color[2] = sm.color[2];
                vm.tri_colors = (run_mesh_vis >= 3 || sm.color_override || static_vis_colors[i].empty()) ? nullptr : static_vis_colors[i].data();
                svm.push_back(vm);
            }
            vis_set_static_meshes(vis, svm);
            std::vector<VisMesh> dvm;
            size_t doff = static_tris.size();
            for (size_t i = 0; i < scene_cfg.dynamic_meshes.size(); ++i) {
                const auto& dm = scene_cfg.dynamic_meshes[i];
                VisMesh vm;
                vm.tris       = init_tris.data() + doff;
                vm.count      = dyn_mesh_tris[i].size();
                vm.color[0]   = dm.color[0]; vm.color[1] = dm.color[1]; vm.color[2] = dm.color[2];
                vm.tri_colors = (run_mesh_vis >= 3 || dm.color_override || dyn_mesh_colors[i].empty()) ? nullptr : dyn_mesh_colors[i].data();
                dvm.push_back(vm);
                doff += dyn_mesh_tris[i].size();
            }
            vis_update_dynamic_meshes(vis, dvm);

            // Build initial AABB list: one per static mesh + one per dynamic mesh
            for (size_t i = 0; i < scene_cfg.static_meshes.size(); ++i) {
                VisAABB va;
                float xmn = std::numeric_limits<float>::max(),    ymn = xmn, zmn = xmn;
                float xmx = std::numeric_limits<float>::lowest(), ymx = xmx, zmx = xmx;
                for (const auto& tri : static_vis_tris[i])
                    for (int k = 0; k < 3; ++k) {
                        xmn=std::min(xmn,tri.v[k][0]); xmx=std::max(xmx,tri.v[k][0]);
                        ymn=std::min(ymn,tri.v[k][1]); ymx=std::max(ymx,tri.v[k][1]);
                        zmn=std::min(zmn,tri.v[k][2]); zmx=std::max(zmx,tri.v[k][2]);
                    }
                va.aabb[0]=xmn; va.aabb[1]=ymn; va.aabb[2]=zmn;
                va.aabb[3]=xmx; va.aabb[4]=ymx; va.aabb[5]=zmx;
                va.color[0]=scene_cfg.static_meshes[i].color[0];
                va.color[1]=scene_cfg.static_meshes[i].color[1];
                va.color[2]=scene_cfg.static_meshes[i].color[2];
                vis_aabbs.push_back(va);
            }
            if (show_fallback_static_aabb) {
                VisAABB va;
                for (int k = 0; k < 6; ++k) va.aabb[k] = static_aabb[k];
                va.color[0] = 0.7f; va.color[1] = 0.7f; va.color[2] = 0.7f;
                vis_aabbs.push_back(va);
            }
            {
                size_t idoff = static_tris.size();
                for (size_t i = 0; i < scene_cfg.dynamic_meshes.size(); ++i) {
                    VisAABB va;
                    float xmn = std::numeric_limits<float>::max(),    ymn = xmn, zmn = xmn;
                    float xmx = std::numeric_limits<float>::lowest(), ymx = xmx, zmx = xmx;
                    const Tri3* base = init_tris.data() + idoff;
                    const size_t nc = dyn_mesh_tris[i].size();
                    for (size_t ji = 0; ji < nc; ++ji)
                        for (int k = 0; k < 3; ++k) {
                            xmn=std::min(xmn,base[ji].v[k][0]); xmx=std::max(xmx,base[ji].v[k][0]);
                            ymn=std::min(ymn,base[ji].v[k][1]); ymx=std::max(ymx,base[ji].v[k][1]);
                            zmn=std::min(zmn,base[ji].v[k][2]); zmx=std::max(zmx,base[ji].v[k][2]);
                        }
                    va.aabb[0]=xmn; va.aabb[1]=ymn; va.aabb[2]=zmn;
                    va.aabb[3]=xmx; va.aabb[4]=ymx; va.aabb[5]=zmx;
                    va.color[0]=scene_cfg.dynamic_meshes[i].color[0];
                    va.color[1]=scene_cfg.dynamic_meshes[i].color[1];
                    va.color[2]=scene_cfg.dynamic_meshes[i].color[2];
                    vis_aabbs.push_back(va);
                    idoff += nc;
                }
            }
            vis_set_aabbs(vis, vis_aabbs);

            // Initial lidar cylinder + cone upload
            {
                std::vector<VisLidar> vlidars;
                for (const auto& robot : scene_cfg.robots)
                    for (const auto& lidar : robot.lidars) {
                        VisLidar vl;
                        vl.pos[0]   = lidar.w_pose.pos.x;
                        vl.pos[1]   = lidar.w_pose.pos.y;
                        vl.pos[2]   = lidar.w_pose.pos.z;
                        vl.fwd[0]   = lidar.w_pose.up.x;
                        vl.fwd[1]   = lidar.w_pose.up.y;
                        vl.fwd[2]   = lidar.w_pose.up.z;
                        vl.up[0]    = lidar.w_pose.forward.x;
                        vl.up[1]    = lidar.w_pose.forward.y;
                        vl.up[2]    = lidar.w_pose.forward.z;
                        vl.vmin     = lidar.vmin;
                        vl.vmax     = lidar.vmax;
                        vl.vnum     = lidar.vnum;
                        vl.range_max = lidar.range_max;
                        vlidars.push_back(vl);
                    }
                vis_set_lidars(vis, vlidars);
                if (scene_cfg.cone_vis_on) vis_set_lidar_cones(vis, vlidars);
            }
        }

        // -----------------------------------------------------------------------
        // 9. Batch loop
        // -----------------------------------------------------------------------
        for (size_t batchIdx = 0; batchIdx < batches.size(); ++batchIdx) {
            if (!active_filters.empty()) {
                bool match = false;
                for (const auto& [ti, bi] : active_filters) {
                    bool run_ok   = (ti < 0 || (int)runIdx   == ti);
                    bool batch_ok = (bi < 0 || (int)batchIdx == bi);
                    if (run_ok && batch_ok) { match = true; break; }
                }
                if (!match) continue;
            }
            const auto& batch_j = batches[batchIdx];

            std::string batch_dir = run_base + "/batch_" + std::to_string(batchIdx);
        ensure_dir(batch_dir);

        // Determine backend(s) for this batch
        std::string batch_backend = cli_backend.empty()
            ? (batch_j.contains("backend") ? batch_j["backend"].get<std::string>() : "grca-cpu")
            : cli_backend;
        auto backends = expand_backends(batch_backend);
        if (backends.empty()) {
            fprintf(stderr, "[WARN] Backend group \"%s\" expanded to no compiled-in backends — skipping batch %zu.\n",
                    batch_backend.c_str(), batchIdx);
            continue;
        }

        // Batch-level lidar sim rate
        float batch_lsr = batch_j.contains("lidar_sim_rate")
            ? batch_j["lidar_sim_rate"].get<float>()
            : lidar_sim_rate_global;

        // Active test for this batch
        std::string active_test = getActiveBatchTest(batch_j);

        // ------------------------------------------------------------------
        // Per-backend run inside this batch
        // ------------------------------------------------------------------
        {
            int _max = -1;
            for (int n : {perf_record_frames_global, pose_record_frames_count_global, hit_dist_record_frames_global})
                if (n > -1) _max = (_max < 0) ? n : std::max(_max, n);
            if (_max < 0) {
                if (backends.size() > 1) {
                    fprintf(stderr,
                            "[ERROR] Multi-backend group cannot run: all record frame counts are -1 — "
                            "the first backend would run indefinitely and the rest would never execute.\n"
                            "        Please select a dedicated backend: grca-cpu, grca-cuda, optix, optix-crti, embree, tinybvh-cpu, tinybvh-gpu, hybrid-cpu, hybrid-gpu.\n");
                    continue;
                }
                fprintf(stderr, "[INFO] All record frame counts are -1 — running indefinitely. Press Ctrl+C to stop.\n");
            }
        }
        std::vector<BatchBackendResult> batch_results;
        for (const auto& be : backends) {

            fprintf(stderr, "\n=== Batch %zu / %s ===\n", batchIdx, be.c_str());

            // Build run config: fresh copy of scene + apply batch settings
            float dummy_lsr   = batch_lsr;
            int   dummy_ahvrf = auto_hit_view_global;
            RunConfig cfg;
            try { cfg = parseConfig(scene_j, dummy_lsr, dummy_ahvrf); }
            catch (const nlohmann::json::exception& e) {
                fprintf(stderr, "[JSON ERROR in parseConfig] %s\n", e.what());
                throw;
            }

            // Apply random_dynamic_test from batch
            if (batch_j.contains("random_dynamic_test")) {
                const auto& rdt = batch_j["random_dynamic_test"];
                cfg.seed = rdt.value("seed", 0);
                cfg.random_dynamic_from_static_aabb = true;
                if (rdt.contains("pos_interpolate_frames"))
                    cfg.random_dynamic_pos_interpolate_frames   = rdt["pos_interpolate_frames"].get<int>();
                if (rdt.contains("rot_interpolate_frames"))
                    cfg.random_dynamic_rot_interpolate_frames   = rdt["rot_interpolate_frames"].get<int>();
                if (rdt.contains("scale_interpolate_frames"))
                    cfg.random_dynamic_scale_interpolate_frames = rdt["scale_interpolate_frames"].get<int>();
                if (rdt.contains("scale_min")) cfg.random_dynamic_scale_min = rdt["scale_min"].get<float>();
                if (rdt.contains("scale_max")) cfg.random_dynamic_scale_max = rdt["scale_max"].get<float>();
                if (rdt.contains("tri_chaos")) {
                    auto& v = rdt["tri_chaos"];
                    cfg.random_tri_chaos = v.is_boolean() ? (v.get<bool>() ? 1 : 0) : v.get<int>();
                }
                if (rdt.contains("robots_share_pose")) cfg.random_robots_share_pose = rdt["robots_share_pose"].get<bool>();
            }

            // Apply per-test options from cur_run_j
            try {
            cfg.visualizer_on = run_vis_on;
            cfg.mesh_vis_on   = run_mesh_vis;
            cfg.cone_vis_on   = run_cone_vis;
            if (cur_run_j.contains("cuda_debug"))
                cfg.debug = cur_run_j["cuda_debug"].get<bool>();
            if (cur_run_j.contains("cuda")) {
                const auto& cj = cur_run_j["cuda"];
                if (cj.contains("debug"))             cfg.debug                  = cj["debug"].get<bool>();
                if (cj.contains("sat_max_sweep_diff"))   cfg.cuda_sat_max_sweep_diff   = cj["sat_max_sweep_diff"].get<int>();
                if (cj.contains("sat_max_channel_diff")) cfg.cuda_sat_max_channel_diff = cj["sat_max_channel_diff"].get<int>();
            }
            if ((be == "optix" || be == "optix-crti") && cur_run_j.contains("optix")) {
                const auto& oj = cur_run_j["optix"];
                if (oj.contains("bvh_rebuild_after_frames")) cfg.optix_rebuild_after_frames = oj["bvh_rebuild_after_frames"].get<int>();
                if (oj.contains("bvh_compaction_on"))        cfg.optix_compaction           = oj["bvh_compaction_on"].get<bool>();
            }

            // CLI overrides (highest priority)
            cfg.backend = be;
            if (!cli_out.empty()) cfg.output = cli_out;
            if (cli_debug)   cfg.debug = true;
            if (cli_refit)   { cfg.optix_rebuild_after_frames = -1; cfg.embree_rebuild_after_frames = -1; cfg.tinybvh_cpu_rebuild_after_frames = -1; cfg.tinybvh_gpu_rebuild_after_frames = -1; }
            if (cli_rebuild) { cfg.optix_rebuild_after_frames =  0; cfg.embree_rebuild_after_frames =  0; cfg.tinybvh_cpu_rebuild_after_frames =  0; cfg.tinybvh_gpu_rebuild_after_frames =  0; }

            // GRCA-CPU packet size
            if (be == "grca-cpu" && cur_run_j.contains("cpu")) {
                const auto& gj = cur_run_j["cpu"];
                if (gj.contains("packet_size")) {
                    int ps = gj["packet_size"].get<int>();
                    if (ps == 1 || ps == 4 || ps == 8 || ps == 16) cfg.grca_cpu_packet_size = ps;
                }
                if (gj.contains("sat_max_sweep_diff"))   cfg.cpu_sat_max_sweep_diff   = gj["sat_max_sweep_diff"].get<int>();
                if (gj.contains("sat_max_channel_diff")) cfg.cpu_sat_max_channel_diff  = gj["sat_max_channel_diff"].get<int>();
            }

            // Embree packet size and BVH sync mode
            if (be == "embree" && cur_run_j.contains("embree")) {
                const auto& ej = cur_run_j["embree"];
                if (ej.contains("packet_size")) {
                    int ps = ej["packet_size"].get<int>();
                    if (ps == 1 || ps == 4 || ps == 8 || ps == 16) cfg.embree_packet_size = ps;
                }
                if (ej.contains("bvh_rebuild_after_frames")) cfg.embree_rebuild_after_frames = ej["bvh_rebuild_after_frames"].get<int>();
                if (ej.contains("bvh_compaction_on"))        cfg.embree_compaction = ej["bvh_compaction_on"].get<bool>();
            }

            // TinyBVH BVH sync mode
            if ((be == "tinybvh-cpu" || be == "tinybvh-gpu") && cur_run_j.contains("tinybvh")) {
                const auto& tj = cur_run_j["tinybvh"];
                if (tj.contains("bvh_rebuild_after_frames")) {
                    cfg.tinybvh_cpu_rebuild_after_frames = tj["bvh_rebuild_after_frames"].get<int>();
                    cfg.tinybvh_gpu_rebuild_after_frames = tj["bvh_rebuild_after_frames"].get<int>();
                }
                if (tj.contains("cpu_rebuild_after_frames")) cfg.tinybvh_cpu_rebuild_after_frames = tj["cpu_rebuild_after_frames"].get<int>();
                if (tj.contains("gpu_rebuild_after_frames")) cfg.tinybvh_gpu_rebuild_after_frames = tj["gpu_rebuild_after_frames"].get<int>();
            }
            } catch (const nlohmann::json::exception& e) {
                fprintf(stderr, "[JSON ERROR in overrides] %s\n", e.what());
                throw;
            }

            cfg.bat_capacity = json_bat_capacity;

            // Recording settings for this run (from cur_run_j)
            int perf_record_frames      = perf_record_frames_global;
            // Hit-dist and pose CSVs are only useful for cross-backend comparison — skip when only 1 backend runs
            int hit_dist_record_frames  = (backends.size() >= 2) ? hit_dist_record_frames_global : -1;
            int pose_record_frames_count = (backends.size() >= 2) ? pose_record_frames_count_global : -1;
            int max_record_frames = -1;
            for (int n : {perf_record_frames, pose_record_frames_count, hit_dist_record_frames})
                if (n > -1) max_record_frames = (max_record_frames < 0) ? n : std::max(max_record_frames, n);

            // Test configs for this batch
            std::optional<OrbitPairsTestCfg> orbit_pairs_test;
            std::optional<LidarOrbitTestCfg> lidar_orbit_test;
            if (active_test == "orbit_pairs_test") orbit_pairs_test = parseOrbitPairsTest(batch_j, cfg);
            if (active_test == "lidar_orbit_test") lidar_orbit_test = parseLidarOrbitTest(batch_j, cfg);

            // Ray / lidar counts
            int totalRays = 0, lidarCount = 0;
            if (lidar_orbit_test) {
                for (const auto& ref : lidar_orbit_test->lidar_candidates) {
                    totalRays += cfg.robots[ref.robot_idx].lidars[ref.lidar_idx].hnum *
                                 cfg.robots[ref.robot_idx].lidars[ref.lidar_idx].vnum;
                    ++lidarCount;
                }
            } else {
                for (const auto& robot : cfg.robots) {
                    lidarCount += (int)robot.lidars.size();
                    for (const auto& lidar : robot.lidars) totalRays += lidar.hnum * lidar.vnum;
                }
            }
            float capture_dt = 1.0f / batch_lsr;

            fprintf(stderr, "Backend      : %s\n", be.c_str());
            fprintf(stderr, "Robots       : %zu  |  Total rays: %d\n", cfg.robots.size(), totalRays);
            fprintf(stderr, "Capture rate : %.2f Hz  (dt = %.4f s)\n", batch_lsr, capture_dt);
            fprintf(stderr, "Static meshes: %zu\n", cfg.static_meshes.size());
            fprintf(stderr, "Dyn meshes   : %zu\n", cfg.dynamic_meshes.size());
            fprintf(stderr, "Output dir   : %s\n", batch_dir.c_str());
            if (auto_hit_view_global >= 0)
                fprintf(stderr, "Auto-capture at frame %d\n", auto_hit_view_global);
            if (cfg.visualizer_on)
                fprintf(stderr, "Press 'o' to write CSV, 'q' to quit.\n\n");

            // World AABB for this run
            float world_aabb[6] = {};
            bool  has_world_aabb = false;
            if (cfg.random_dynamic_from_static_aabb && has_static_aabb) {
                std::copy(static_aabb, static_aabb + 6, world_aabb);
                has_world_aabb = true;
            }

            double perf_init_cpu_ms = 0.0;

            // Pre-allocate triangle buffer (static portion fixed; dynamic suffix updated each frame)
            size_t total_dyn_tris = 0;
            for (const auto& v : dyn_mesh_tris) total_dyn_tris += v.size();
            std::vector<Tri3> tris(static_tris.size() + total_dyn_tris);
            std::copy(static_tris.begin(), static_tris.end(), tris.begin());

            // ------------------------------------------------------------------
            // Visualizer
            // ------------------------------------------------------------------
            bool running = true;
            // Visualizer is now persistent per test file, not per batch

            // ------------------------------------------------------------------
            // Per-run state
            // ------------------------------------------------------------------
            std::mt19937 teleport_rng;
            if (has_world_aabb) teleport_rng.seed(cfg.seed);
            // Four independent xoshiro streams — OoO engine sees 4 chains of depth 2
            // instead of 2 chains of depth 4, doubling RNG throughput vs two streams.
            XoShiro128Plus cr0, cr1, cr2, cr3;
            if (cfg.random_tri_chaos > 0) {
                cr0.seed((uint64_t)cfg.seed ^ 0xc0ffee01deadULL);
                cr1.seed((uint64_t)cfg.seed ^ 0xdeadbeef1234ULL);
                cr2.seed((uint64_t)cfg.seed ^ 0xabad1dea5678ULL);
                cr3.seed((uint64_t)cfg.seed ^ 0x1234567890abULL);
            }

            // When robots_share_pose, randomise each robot's lidar local orientations once at
            // batch start so all lidars face unique directions from the shared emitter position.
            // Uses a separate RNG (seed XOR'd) so it does not disturb the teleport sequence.
            if (cfg.random_robots_share_pose) {
                std::mt19937 lidar_ori_rng(static_cast<uint32_t>(cfg.seed) ^ 0xA5A5A5A5u);
                std::normal_distribution<float> nd(0.f, 1.f);
                for (auto& robot : cfg.robots) {
                    for (auto& lidar : robot.lidars) {
                        // Random unit forward via Gaussian sampling on sphere
                        float fx, fy, fz, fn;
                        do {
                            fx = nd(lidar_ori_rng); fy = nd(lidar_ori_rng); fz = nd(lidar_ori_rng);
                            fn = std::sqrt(fx*fx + fy*fy + fz*fz);
                        } while (fn < 1e-6f);
                        fx /= fn; fy /= fn; fz /= fn;
                        // Perpendicular up via Gram-Schmidt
                        float rx = (std::abs(fy) < 0.9f) ? 0.f : 1.f;
                        float ry = (std::abs(fy) < 0.9f) ? 1.f : 0.f;
                        float rz = 0.f;
                        float d = rx*fx + ry*fy + rz*fz;
                        float ux = rx - d*fx, uy = ry - d*fy, uz = rz - d*fz;
                        float un = std::sqrt(ux*ux + uy*uy + uz*uz);
                        ux /= un; uy /= un; uz /= un;
                        lidar.l_pose.forward = {fx, fy, fz};
                        lidar.l_pose.up      = {ux, uy, uz};
                        lidar.l_pose.right   = {fy*uz - fz*uy, fz*ux - fx*uz, fx*uy - fy*ux};
                    }
                }
            }

            struct TeleportState {
                float prev[3]={}, next[3]={};
                float prev_euler[3]={}, next_euler[3]={};
                float prev_scale[3]={1.f,1.f,1.f}, next_scale[3]={1.f,1.f,1.f};
                int   interp_frame=0;
                bool  initialized=false;
            };
            std::vector<TeleportState> robot_ts(cfg.robots.size());
            std::vector<TeleportState> dynmesh_ts(cfg.dynamic_meshes.size());
            std::vector<TeleportState> orbit_rot_ts(cfg.dynamic_meshes.size());

            std::vector<bool> robot_mask(cfg.robots.size(), true);
            std::vector<std::vector<bool>> robot_lidar_mask(cfg.robots.size());
            for (size_t i = 0; i < cfg.robots.size(); ++i)
                robot_lidar_mask[i].assign(cfg.robots[i].lidars.size(), true);
            std::vector<bool> dynmesh_mask(cfg.dynamic_meshes.size(), true);

            // Apply candidates filter from random_dynamic_test
            if (batch_j.contains("random_dynamic_test")) {
                const auto& rdt = batch_j["random_dynamic_test"];
                if (rdt.contains("candidates") && rdt["candidates"].is_array()) {
                    std::fill(robot_mask.begin(),  robot_mask.end(),  false);
                    std::fill(dynmesh_mask.begin(), dynmesh_mask.end(), false);
                    for (const auto& c : rdt["candidates"]) {
                        if (!c.is_string()) continue;
                        std::string s = c.get<std::string>();
                        if (s == "robots") {
                            std::fill(robot_mask.begin(), robot_mask.end(), true);
                        } else if (s == "dynamic_meshes") {
                            std::fill(dynmesh_mask.begin(), dynmesh_mask.end(), true);
                        } else if (s.find("robots[") == 0) {
                            auto p0 = s.find('['), p1 = s.find(']');
                            if (p0 != std::string::npos && p1 != std::string::npos) {
                                int idx = std::stoi(s.substr(p0+1, p1-p0-1));
                                if (idx >= 0 && idx < (int)robot_mask.size()) robot_mask[idx] = true;
                            }
                        } else if (s.find("dynamic_meshes[") == 0) {
                            auto p0 = s.find('['), p1 = s.find(']');
                            if (p0 != std::string::npos && p1 != std::string::npos) {
                                int idx = std::stoi(s.substr(p0+1, p1-p0-1));
                                if (idx >= 0 && idx < (int)dynmesh_mask.size()) dynmesh_mask[idx] = true;
                            }
                        }
                    }
                }
            }

            auto pick_dest = [&]() {
                std::uniform_real_distribution<float> dx(world_aabb[0],world_aabb[3]);
                std::uniform_real_distribution<float> dy(world_aabb[1],world_aabb[4]);
                std::uniform_real_distribution<float> dz(world_aabb[2],world_aabb[5]);
                return std::array<float,3>{dx(teleport_rng), dy(teleport_rng), dz(teleport_rng)};
            };
            auto pick_angle = [&]() {
                std::uniform_real_distribution<float> d(0.f, 2.f * 3.14159265f);
                return d(teleport_rng);
            };
            auto pick_scale3 = [&](float* out) {
                // Log-uniform per axis: equal representation per decade, independent stretching per axis
                const float lo   = std::max(cfg.random_dynamic_scale_min, 1e-9f);
                const float lslo3 = logf(lo);
                const float lsr3  = logf(std::max(cfg.random_dynamic_scale_max, lo)) - lslo3;
                std::uniform_real_distribution<float> u(0.f, 1.f);
                for (int e = 0; e < 3; ++e) out[e] = expf(lslo3 + u(teleport_rng) * lsr3);
            };

            float orbit_angle = 0.0f;
            float orbit_pairs_angle = 0.0f;
            std::mt19937 orbit_rng, orbit_pairs_rng;
            if (lidar_orbit_test)  orbit_rng.seed(lidar_orbit_test->seed);
            if (orbit_pairs_test)  orbit_pairs_rng.seed(orbit_pairs_test->seed);

            // Perf CSV accumulators
            double sum_fps=0, sum_rtf=0, sum_lidar_sim=0;
            double sum_bvh_rebuild_gpu=0, sum_bvh_refit_gpu=0, sum_bvh_rebuild_cpu=0, sum_bvh_refit_cpu=0;
            double sum_trace=0, sum_trace_cpu=0, sum_sync_rb=0, sum_dyn_up=0, sum_fillhits=0, sum_rb=0;
            double sum_lidar_sim_gpu=0, sum_total_cpu=0;
            double min_lidar_sim_cpu=1e30, max_lidar_sim_cpu=0;
            double min_lidar_sim_gpu=1e30, max_lidar_sim_gpu=0;
            int perf_frame_count = 0;

            std::ofstream perf_csv_file;

            // Per-run CSV paths (created lazily inside the loop)
            std::string perf_csv_name     = make_timestamped_csv_name("perf",     cfg, batch_dir);
            std::string pose_csv_name;
            std::string hit_dist_csv_name;
            std::unique_ptr<std::ofstream> pose_csv_out;
            std::unique_ptr<std::ofstream> hit_dist_csv_out;

            double prev_frame_real_time = -1.0;
            int frame = 0;
            float sim_time = 0.0f;
            std::vector<GrcaHit> latest_hits;
            int   last_vis_frame = 0;
            float last_vis_sim_time = 0.0f;
            auto next_sim_wall_time = clk::now();

            // ------------------------------------------------------------------
            // Pre-teleport: scatter all teleporting objects to random starting
            // positions before frame 0 (uses the batch seed, so reproducible).
            // Leaving objects at origin would be a pathological worst case.
            // ------------------------------------------------------------------
            if (has_world_aabb) {
                int pf = cfg.random_dynamic_pos_interpolate_frames;
                int rf = cfg.random_dynamic_rot_interpolate_frames;
                int sf = cfg.random_dynamic_scale_interpolate_frames;
                bool pe = pf >= 0, re = rf >= 0, se = sf >= 0;
                if (pe || re || se) {
                    for (size_t i = 0; i < cfg.dynamic_meshes.size(); ++i) {
                        if (!dynmesh_mask[i]) continue;
                        auto& ts = dynmesh_ts[i];
                        if (pe) { auto d = pick_dest(); for (int e=0;e<3;++e) { ts.prev[e]=d[e]; ts.next[e]=d[e]; } }
                        else    { ts.prev[0]=cfg.dynamic_meshes[i].w_pose.pos.x;
                                  ts.prev[1]=cfg.dynamic_meshes[i].w_pose.pos.y;
                                  ts.prev[2]=cfg.dynamic_meshes[i].w_pose.pos.z;
                                  for (int e=0;e<3;++e) ts.next[e]=ts.prev[e]; }
                        if (re) { for (int e=0;e<3;++e) { ts.prev_euler[e]=pick_angle(); ts.next_euler[e]=ts.prev_euler[e]; } }
                        else    { for (int e=0;e<3;++e) { ts.prev_euler[e]=0.f; ts.next_euler[e]=0.f; } }
                        if (se) pick_scale3(ts.next_scale);
                        else for (int e=0;e<3;++e) ts.next_scale[e]=cfg.dynamic_meshes[i].scale;
                        for (int e=0;e<3;++e) ts.prev_scale[e]=ts.next_scale[e];
                        ts.interp_frame = 0; ts.initialized = true;
                    }
                    for (size_t ri = 0; ri < cfg.robots.size(); ++ri) {
                        if (!robot_mask[ri]) continue;
                        // When all robots share a pose, only robot_ts[0] needs initialising
                        if (cfg.random_robots_share_pose && ri > 0) continue;
                        auto& ts = robot_ts[ri];
                        if (pe) { auto d = pick_dest(); for (int e=0;e<3;++e) { ts.prev[e]=d[e]; ts.next[e]=d[e]; } }
                        else    { ts.prev[0]=cfg.robots[ri].w_pose.pos.x;
                                  ts.prev[1]=cfg.robots[ri].w_pose.pos.y;
                                  ts.prev[2]=cfg.robots[ri].w_pose.pos.z;
                                  for (int e=0;e<3;++e) ts.next[e]=ts.prev[e]; }
                        if (re) { for (int e=0;e<3;++e) { ts.prev_euler[e]=pick_angle(); ts.next_euler[e]=ts.prev_euler[e]; } }
                        else    { for (int e=0;e<3;++e) { ts.prev_euler[e]=0.f; ts.next_euler[e]=0.f; } }
                        ts.interp_frame = 0; ts.initialized = true;
                    }
                }
            }

            // Rebuild init_tris dynamic portion from pre-teleported positions
            // so the backend BVH is built from a realistic starting configuration.
            if (has_world_aabb && cfg.random_dynamic_pos_interpolate_frames >= 0) {
                size_t doff = static_tris.size();
                for (size_t i = 0; i < cfg.dynamic_meshes.size(); ++i) {
                    float tx = dynmesh_ts[i].prev[0];
                    float ty = dynmesh_ts[i].prev[1];
                    float tz = dynmesh_ts[i].prev[2];
                    const float* sc3 = dynmesh_ts[i].prev_scale;
                    for (size_t ji = 0; ji < dyn_mesh_tris[i].size(); ++ji) {
                        const auto& src = dyn_mesh_tris[i][ji];
                        for (int k = 0; k < 3; ++k) {
                            init_tris[doff + ji].v[k][0] = src.v[k][0] * sc3[0] + tx;
                            init_tris[doff + ji].v[k][1] = src.v[k][1] * sc3[1] + ty;
                            init_tris[doff + ji].v[k][2] = src.v[k][2] * sc3[2] + tz;
                        }
                    }
                    doff += dyn_mesh_tris[i].size();
                }
            }

            // ------------------------------------------------------------------
            // Backend init (after pre-teleport so BVH sees randomised positions)
            // ------------------------------------------------------------------
            if (be == "grca-cpu") {
                FrameTimings initT = {};
                cpuState = init_cpu(cfg, "grca-cpu", initT);
                if (!cpuState) { fprintf(stderr, "init_cpu failed\n"); continue; }
                perf_init_cpu_ms = initT.ms_init_cpu;
                fprintf(stderr, "--- GRCA_CPU init timings ---\n");
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), init_tris.size() - static_tris.size(), initT, session_label);
            }
#ifdef HAVE_CUDA
            size_t vram_before_init = 0, vram_total_gpu = 0;
            if (be == "grca-cuda" || be == "optix" || be == "optix-crti") {
                size_t f = 0;
                cudaMemGetInfo(&f, &vram_total_gpu);
                vram_before_init = vram_total_gpu - f;
            }
            if (be == "grca-cuda") {
                FrameTimings initT = {};
                cudaState = init_cuda(cfg, init_tris, static_tris.size(), initT);
                if (!cudaState) { fprintf(stderr, "init_cuda failed\n"); continue; }
                perf_init_cpu_ms = initT.ms_init_cpu;
                fprintf(stderr, "--- GRCA CUDA init timings ---\n");
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), init_tris.size() - static_tris.size(), initT, session_label);
            }
#endif
#ifdef HAVE_OPTIX
            if (be == "optix" || be == "optix-crti") {
                FrameTimings initT = {};
                optixState = init_optix(cfg, init_tris, static_tris.size(), initT);
                if (!optixState) { fprintf(stderr, "init_optix failed\n"); continue; }
                perf_init_cpu_ms = initT.ms_init_cpu;
                fprintf(stderr, "--- OptiX init timings ---\n");
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), init_tris.size() - static_tris.size(), initT, session_label);
            }
#endif
#ifdef HAVE_CUDA
            if ((be == "grca-cuda" || be == "optix" || be == "optix-crti") && vram_total_gpu > 0) {
                size_t curFree = 0, curTotal = 0;
                cudaMemGetInfo(&curFree, &curTotal);
                double persistMB = (double)(curTotal - curFree) / (1024.0 * 1024.0);
                double allocMB   = persistMB - (double)vram_before_init / (1024.0 * 1024.0);
                fprintf(stderr, "[VRAM] allocated: %.1f MB  persistent: %.1f MB / %.1f MB total\n",
                        allocMB, persistMB, (double)curTotal / (1024.0 * 1024.0));
            }
#endif
#ifdef HAVE_EMBREE
            if (be == "embree") {
                FrameTimings initT = {};
                embreeState = init_embree(cfg, init_tris, static_tris.size(), initT);
                if (!embreeState) { fprintf(stderr, "init_embree failed\n"); continue; }
                perf_init_cpu_ms = initT.ms_init_cpu;
                fprintf(stderr, "--- Embree init timings ---\n");
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), init_tris.size() - static_tris.size(), initT, session_label);
            }
#endif
#ifdef HAVE_TINYBVH
            if (be == "tinybvh-cpu") {
                FrameTimings initT = {};
                tinybvhCpuState = init_tinybvh_cpu(cfg, init_tris, static_tris.size(), initT);
                if (!tinybvhCpuState) { fprintf(stderr, "init_tinybvh_cpu failed\n"); continue; }
                perf_init_cpu_ms = initT.ms_init_cpu;
                fprintf(stderr, "--- TinyBVH-CPU init timings ---\n");
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), init_tris.size() - static_tris.size(), initT, session_label);
            }
            if (be == "tinybvh-gpu") {
                FrameTimings initT = {};
                tinybvhGpuState = init_tinybvh_gpu(cfg, init_tris, static_tris.size(), initT);
                if (!tinybvhGpuState) { fprintf(stderr, "init_tinybvh_gpu failed\n"); continue; }
                perf_init_cpu_ms = initT.ms_init_cpu;
                fprintf(stderr, "--- TinyBVH-GPU init timings ---\n");
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), init_tris.size() - static_tris.size(), initT, session_label);
            }
#endif
#ifdef HAVE_EMBREE
            if (be == "hybrid-cpu") {
                FrameTimings initT = {};
                hybridCpuState = init_hybrid_cpu(cfg, static_tris, initT);
                if (!hybridCpuState) { fprintf(stderr, "init_hybrid_cpu failed\n"); continue; }
                perf_init_cpu_ms = initT.ms_init_cpu;
                fprintf(stderr, "--- Hybrid-CPU init timings ---\n");
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), init_tris.size() - static_tris.size(), initT, session_label);
            }
#endif
#if defined(HAVE_OPTIX) && defined(HAVE_CUDA)
            if (be == "hybrid-gpu") {
                size_t f = 0;
                cudaMemGetInfo(&f, &vram_total_gpu);
                vram_before_init = vram_total_gpu - f;
                FrameTimings initT = {};
                std::vector<Tri3> dyn_init_tris(init_tris.begin() + static_cast<ptrdiff_t>(static_tris.size()), init_tris.end());
                hybridGpuState = init_hybrid_gpu(cfg, static_tris, dyn_init_tris, initT);
                if (!hybridGpuState) { fprintf(stderr, "init_hybrid_gpu failed\n"); continue; }
                perf_init_cpu_ms = initT.ms_init_cpu;
                fprintf(stderr, "--- Hybrid-GPU init timings ---\n");
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), init_tris.size() - static_tris.size(), initT, session_label);
                if (vram_total_gpu > 0) {
                    size_t curFree = 0, curTotal = 0;
                    cudaMemGetInfo(&curFree, &curTotal);
                    double persistMB = (double)(curTotal - curFree) / (1024.0 * 1024.0);
                    double allocMB   = persistMB - (double)vram_before_init / (1024.0 * 1024.0);
                    fprintf(stderr, "[VRAM] allocated: %.1f MB  persistent: %.1f MB / %.1f MB total\n",
                            allocMB, persistMB, (double)curTotal / (1024.0 * 1024.0));
                }
            }
#endif



            // ------------------------------------------------------------------
            // Frame loop
            // ------------------------------------------------------------------
            if (perf_record_frames > -1) {
                perf_csv_file.open(perf_csv_name);
                write_perf_csv_header(perf_csv_file, batch_lsr);
                perf_csv_file << "frame,ms_lidar_sim_cpu,ms_dynamic_upload,ms_bvh_rebuild_cpu,ms_bvh_refit_cpu,ms_trace_cpu,ms_sync_readback,ms_readback_cpu,ms_fillhits,"
                                "ms_lidar_sim_gpu,ms_bvh_rebuild_gpu,ms_bvh_refit_gpu,"
                                "ms_init_cpu,sim_frame_rate,real_time_factor,ms_total_cpu,mb_readback,hit_count\n";
                perf_csv_file.flush();
            }

#ifdef HAVE_CUDA
            size_t batch_max_vram = 0;
#endif
            double batch_max_ram_mb = 0.0;
            int bat_min = INT_MAX, bat_max = -1;
            long long bat_sum = 0;
            int bat_frame_count = 0;
            int bat_capacity = -1;
            int early_t_min = INT_MAX, early_t_max = -1;
            long long early_t_sum = 0;
            long long rtic_sum = 0;
            int rtic_min = INT_MAX, rtic_max = -1;
            while (running) {
                if (cfg.visualizer_on && vis) {
                    auto now_wall = clk::now();
                    if (now_wall < next_sim_wall_time) {
                        vis_update(vis, latest_hits, cfg, last_vis_frame, last_vis_sim_time);
                        if (!vis_running(vis)) running = false;
                        continue;
                    }
                    next_sim_wall_time = now_wall +
                        std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(capture_dt));
                }

                auto tStart = clk::now();

                // Auto-quit
                if (max_record_frames > -1 && frame >= max_record_frames) {
                    fprintf(stdout, "[GRCA] Reached max_record_frames=%d. Exiting.\n", max_record_frames);
                    break;
                }

                // Pose recording (before pose updates so frame 0 reflects initial state)
                if (pose_record_frames_count > -1 && frame < pose_record_frames_count && pose_record_global) {
                    if (pose_csv_name.empty()) {
                        pose_csv_name = make_timestamped_csv_name("pose", cfg, batch_dir);
                        pose_csv_out  = std::make_unique<std::ofstream>(pose_csv_name);
                        *pose_csv_out << "frame,type,index,px,py,pz,fx,fy,fz,ux,uy,uz,scale\n";
                    }
                    if (pose_csv_out && pose_csv_out->is_open()) {
                        for (const auto& cand : pose_record_global->candidates) {
                            auto& pco = *pose_csv_out;
                            if (cand.first == "robot") {
                                const auto& r = cfg.robots[cand.second].w_pose;
                                pco << frame << ",robot," << cand.second << ","
                                    << r.pos.x << "," << r.pos.y << "," << r.pos.z << ","
                                    << r.forward.x << "," << r.forward.y << "," << r.forward.z << ","
                                    << r.up.x << "," << r.up.y << "," << r.up.z << ",1.0\n";
                            } else if (cand.first == "dynamic_mesh") {
                                const auto& dm = cfg.dynamic_meshes[cand.second];
                                pco << frame << ",dynamic_mesh," << cand.second << ","
                                    << dm.w_pose.pos.x << "," << dm.w_pose.pos.y << "," << dm.w_pose.pos.z << ","
                                    << dm.w_pose.forward.x << "," << dm.w_pose.forward.y << "," << dm.w_pose.forward.z << ","
                                    << dm.w_pose.up.x << "," << dm.w_pose.up.y << "," << dm.w_pose.up.z << ","
                                    << dm.scale << "\n";
                            } else if (cand.first == "static_mesh") {
                                const auto& sm = cfg.static_meshes[cand.second];
                                pco << frame << ",static_mesh," << cand.second << ","
                                    << sm.w_pose.pos.x << "," << sm.w_pose.pos.y << "," << sm.w_pose.pos.z << ","
                                    << sm.w_pose.forward.x << "," << sm.w_pose.forward.y << "," << sm.w_pose.forward.z << ","
                                    << sm.w_pose.up.x << "," << sm.w_pose.up.y << "," << sm.w_pose.up.z << ","
                                    << sm.scale << "\n";
                            }
                        }
                    }
                }

                // ------------------------------------------------------------------
                // Dynamic mesh transform
                // ------------------------------------------------------------------
                size_t dyn_write = static_tris.size();
                if (cfg.random_tri_chaos > 0 && (has_world_aabb || cfg.random_tri_chaos == 2)) {
                    // Hoist scale range. Log-uniform distribution: equal representation per decade.
                    const float chaos_scale_lo = cfg.random_dynamic_scale_min;
                    const float chaos_scale_w  = cfg.random_dynamic_scale_max - cfg.random_dynamic_scale_min;
                    const float lslo = logf(std::max(chaos_scale_lo, 1e-9f));
                    const float lsr  = logf(std::max(chaos_scale_lo + chaos_scale_w, 1e-9f)) - lslo;
                    size_t doff = static_tris.size();
                    for (size_t i = 0; i < cfg.dynamic_meshes.size(); ++i) {
                        if (!dynmesh_mask[i]) {
                            // Masked out — write at original pose, no chaos
                            const auto& dm = cfg.dynamic_meshes[i];
                            for (size_t ji = 0; ji < dyn_mesh_tris[i].size(); ++ji) {
                                const auto& src = dyn_mesh_tris[i][ji];
                                auto& dst = tris[doff + ji];
                                for (int k = 0; k < 3; ++k) {
                                    float vx=src.v[k][0]*dm.scale, vy=src.v[k][1]*dm.scale, vz=src.v[k][2]*dm.scale;
                                    dst.v[k][0] = dm.w_pose.right.x*vx + dm.w_pose.up.x*vy + dm.w_pose.forward.x*vz + dm.w_pose.pos.x;
                                    dst.v[k][1] = dm.w_pose.right.y*vx + dm.w_pose.up.y*vy + dm.w_pose.forward.y*vz + dm.w_pose.pos.y;
                                    dst.v[k][2] = dm.w_pose.right.z*vx + dm.w_pose.up.z*vy + dm.w_pose.forward.z*vz + dm.w_pose.pos.z;
                                }
                            }
                            doff += dyn_mesh_tris[i].size();
                            continue;
                        }
                        // Compute mesh-level pos + scale + rotation teleport
                        float tx = cfg.dynamic_meshes[i].w_pose.pos.x;
                        float ty = cfg.dynamic_meshes[i].w_pose.pos.y;
                        float tz = cfg.dynamic_meshes[i].w_pose.pos.z;
                        float cur_scale[3] = {cfg.dynamic_meshes[i].scale, cfg.dynamic_meshes[i].scale, cfg.dynamic_meshes[i].scale};
                        bool use_obb = (cfg.random_tri_chaos == 2 && i < dyn_mesh_aabbs.size());
                        Pose meshRot; meshRot.forward = {0,0,1}; meshRot.up = {0,1,0};
                        {
                            int pf = cfg.random_dynamic_pos_interpolate_frames;
                            int rf = cfg.random_dynamic_rot_interpolate_frames;
                            int sf = cfg.random_dynamic_scale_interpolate_frames;
                            bool pe = pf >= 0, re = rf >= 0, se = sf >= 0;
                            if (pe || re || se) {
                                int tot = 0;
                                if (pe) tot = std::max(tot, pf);
                                if (re) tot = std::max(tot, rf);
                                if (se) tot = std::max(tot, sf);
                                auto& ts = dynmesh_ts[i];
                                if (!ts.initialized || ts.interp_frame >= tot) {
                                    if (!ts.initialized) {
                                        ts.prev[0]=cfg.dynamic_meshes[i].w_pose.pos.x;
                                        ts.prev[1]=cfg.dynamic_meshes[i].w_pose.pos.y;
                                        ts.prev[2]=cfg.dynamic_meshes[i].w_pose.pos.z;
                                        for (int e=0;e<3;++e) ts.prev_scale[e] = cfg.dynamic_meshes[i].scale;
                                        for (int e=0;e<3;++e) ts.prev_euler[e] = 0.f;
                                    } else {
                                        for (int e=0;e<3;++e) ts.prev[e]=ts.next[e];
                                        for (int e=0;e<3;++e) ts.prev_scale[e]=ts.next_scale[e];
                                        for (int e=0;e<3;++e) ts.prev_euler[e]=ts.next_euler[e];
                                    }
                                    if (pe) { auto d=pick_dest(); ts.next[0]=d[0]; ts.next[1]=d[1]; ts.next[2]=d[2]; }
                                    else    for (int e=0;e<3;++e) ts.next[e]=ts.prev[e];
                                    if (re) { for (int e=0;e<3;++e) ts.next_euler[e]=pick_angle(); }
                                    else    for (int e=0;e<3;++e) ts.next_euler[e]=ts.prev_euler[e];
                                    if (se) pick_scale3(ts.next_scale);
                                    else for (int e=0;e<3;++e) ts.next_scale[e]=ts.prev_scale[e];
                                    ts.interp_frame = 0; ts.initialized = true;
                                }
                                if (pe) {
                                    float pa = (pf==0)?1.f:std::min(1.f,float(ts.interp_frame)/float(pf));
                                    tx = ts.prev[0]*(1.f-pa)+ts.next[0]*pa;
                                    ty = ts.prev[1]*(1.f-pa)+ts.next[1]*pa;
                                    tz = ts.prev[2]*(1.f-pa)+ts.next[2]*pa;
                                }
                                if (use_obb && re) {
                                    float ra = (rf==0)?1.f:std::min(1.f,float(ts.interp_frame)/float(rf));
                                    float ex = ts.prev_euler[0]*(1.f-ra)+ts.next_euler[0]*ra;
                                    float ey = ts.prev_euler[1]*(1.f-ra)+ts.next_euler[1]*ra;
                                    float ez = ts.prev_euler[2]*(1.f-ra)+ts.next_euler[2]*ra;
                                    rotationFromEuler(ex, ey, ez, meshRot);
                                }
                                if (se) {
                                    float sa = (sf==0)?1.f:std::min(1.f,float(ts.interp_frame)/float(sf));
                                    for (int e=0;e<3;++e) cur_scale[e]=ts.prev_scale[e]*(1.f-sa)+ts.next_scale[e]*sa;
                                }
                                ts.interp_frame++;
                            }
                        }
                        // Precompute per-mesh AABB ranges for direct float scaling (no distributions)
                        const float* ab = use_obb ? dyn_mesh_aabbs[i].data() : world_aabb;
                        const float abx0=ab[0], aby0=ab[1], abz0=ab[2];
                        const float abxw=ab[3]-ab[0], abyw=ab[4]-ab[1], abzw=ab[5]-ab[2];
                        // OBB column vectors: each local axis scaled independently
                        const float ob0x=meshRot.right.x*cur_scale[0],   ob0y=meshRot.right.y*cur_scale[0],   ob0z=meshRot.right.z*cur_scale[0];
                        const float ob1x=meshRot.up.x*cur_scale[1],      ob1y=meshRot.up.y*cur_scale[1],      ob1z=meshRot.up.z*cur_scale[1];
                        const float ob2x=meshRot.forward.x*cur_scale[2], ob2y=meshRot.forward.y*cur_scale[2], ob2z=meshRot.forward.z*cur_scale[2];
                        const auto& cents = dyn_mesh_centroids[i];
                        const size_t ntris = dyn_mesh_tris[i].size();
                        // Inner loop split by use_obb to remove per-tri branch
                        // 4 independent xoshiro streams — cr0: px+sx+qx  cr1: py+sy+qy  cr2: pz+sz+qz  cr3: qw
                        // Per-axis log-uniform scale: independent sx/sy/sz give non-uniform stretching.
#define CHAOS_TRI(src, dst, cx, cy, cz, rpx_, rpy_, rpz_, sx_, sy_, sz_, rng0, rng1, rng2, rng3) \
                        {                                                           \
                            const float rpx=(rpx_), rpy=(rpy_), rpz=(rpz_);       \
                            float qx=rng0.f11(), qy=rng1.f11(), qz=rng2.f11(), qw=rng3.f11(); \
                            const float qi=fast_rsqrt(qx*qx+qy*qy+qz*qz+qw*qw); \
                            qx*=qi; qy*=qi; qz*=qi; qw*=qi;                       \
                            const float r00=1.f-2.f*(qy*qy+qz*qz), r01=2.f*(qx*qy-qz*qw), r02=2.f*(qx*qz+qy*qw); \
                            const float r10=2.f*(qx*qy+qz*qw),     r11=1.f-2.f*(qx*qx+qz*qz), r12=2.f*(qy*qz-qx*qw); \
                            const float r20=2.f*(qx*qz-qy*qw),     r21=2.f*(qy*qz+qx*qw),     r22=1.f-2.f*(qx*qx+qy*qy); \
                            const float vx0=(src.v[0][0]-cx)*(sx_), vy0=(src.v[0][1]-cy)*(sy_), vz0=(src.v[0][2]-cz)*(sz_); \
                            const float vx1=(src.v[1][0]-cx)*(sx_), vy1=(src.v[1][1]-cy)*(sy_), vz1=(src.v[1][2]-cz)*(sz_); \
                            const float vx2=(src.v[2][0]-cx)*(sx_), vy2=(src.v[2][1]-cy)*(sy_), vz2=(src.v[2][2]-cz)*(sz_); \
                            dst.v[0][0]=r00*vx0+r01*vy0+r02*vz0+rpx; dst.v[0][1]=r10*vx0+r11*vy0+r12*vz0+rpy; dst.v[0][2]=r20*vx0+r21*vy0+r22*vz0+rpz; \
                            dst.v[1][0]=r00*vx1+r01*vy1+r02*vz1+rpx; dst.v[1][1]=r10*vx1+r11*vy1+r12*vz1+rpy; dst.v[1][2]=r20*vx1+r21*vy1+r22*vz1+rpz; \
                            dst.v[2][0]=r00*vx2+r01*vy2+r02*vz2+rpx; dst.v[2][1]=r10*vx2+r11*vy2+r12*vz2+rpy; dst.v[2][2]=r20*vx2+r21*vy2+r22*vz2+rpz; \
                        }
                        if (use_obb) {
                            for (size_t ji = 0; ji < ntris; ++ji) {
                                if (ji + 8 < ntris) {
                                    __builtin_prefetch(&dyn_mesh_tris[i][ji+8], 0, 1);
                                    __builtin_prefetch(&cents[ji+8],            0, 1);
                                }
                                const auto& src = dyn_mesh_tris[i][ji];
                                auto& dst = tris[doff + ji];
                                const float cx=cents[ji][0], cy=cents[ji][1], cz=cents[ji][2];
                                const float lx=cr0.frange(abx0,abxw), ly=cr1.frange(aby0,abyw), lz=cr2.frange(abz0,abzw);
                                const float sx=expf(lslo+cr0.f01()*lsr), sy=expf(lslo+cr1.f01()*lsr), sz=expf(lslo+cr2.f01()*lsr);
                                CHAOS_TRI(src, dst, cx, cy, cz,
                                    ob0x*lx+ob1x*ly+ob2x*lz+tx,
                                    ob0y*lx+ob1y*ly+ob2y*lz+ty,
                                    ob0z*lx+ob1z*ly+ob2z*lz+tz,
                                    sx, sy, sz, cr0, cr1, cr2, cr3)
                            }
                        } else {
                            for (size_t ji = 0; ji < ntris; ++ji) {
                                if (ji + 8 < ntris) {
                                    __builtin_prefetch(&dyn_mesh_tris[i][ji+8], 0, 1);
                                    __builtin_prefetch(&cents[ji+8],            0, 1);
                                }
                                const auto& src = dyn_mesh_tris[i][ji];
                                auto& dst = tris[doff + ji];
                                const float cx=cents[ji][0], cy=cents[ji][1], cz=cents[ji][2];
                                const float px=cr0.frange(abx0,abxw), py=cr1.frange(aby0,abyw), pz=cr2.frange(abz0,abzw);
                                const float sx=expf(lslo+cr0.f01()*lsr), sy=expf(lslo+cr1.f01()*lsr), sz=expf(lslo+cr2.f01()*lsr);
                                CHAOS_TRI(src, dst, cx, cy, cz, px, py, pz,
                                    sx, sy, sz, cr0, cr1, cr2, cr3)
                            }
                        }
#undef CHAOS_TRI
                        doff += dyn_mesh_tris[i].size();
                    }
                } else if (orbit_pairs_test && active_test == "orbit_pairs_test") {
                    int rot_frames = orbit_pairs_test->dyn_mesh_rot_interpolate_frames;
                    size_t doff = static_tris.size();
                    for (size_t mi = 0; mi < cfg.dynamic_meshes.size(); ++mi) {
                        const auto& dm = cfg.dynamic_meshes[mi];
                        float euler[3] = {};
                        if (rot_frames >= 0) {
                            auto& ts = orbit_rot_ts[mi];
                            if (!ts.initialized || ts.interp_frame >= std::max(rot_frames, 1)) {
                                if (ts.initialized)
                                    for (int e=0;e<3;++e) ts.prev_euler[e] = ts.next_euler[e];
                                else
                                    ts.prev_euler[0] = ts.prev_euler[1] = ts.prev_euler[2] = 0.f;
                                std::uniform_real_distribution<float> ad(0.f, 2.f*3.14159265f);
                                for (int e=0;e<3;++e) ts.next_euler[e] = ad(orbit_pairs_rng);
                                ts.interp_frame = 0; ts.initialized = true;
                            }
                            float ra = (rot_frames==0) ? 1.f : std::min(1.f, float(ts.interp_frame)/float(rot_frames));
                            for (int e=0;e<3;++e) euler[e] = ts.prev_euler[e]*(1.f-ra) + ts.next_euler[e]*ra;
                            ts.interp_frame++;
                        }
                        Pose rot; rotationFromEuler(euler[0],euler[1],euler[2],rot);
                        float tx=dm.w_pose.pos.x, ty=dm.w_pose.pos.y, tz=dm.w_pose.pos.z;
                        for (size_t ji = 0; ji < dyn_mesh_tris[mi].size(); ++ji) {
                            const auto& src = dyn_mesh_tris[mi][ji];
                            auto& dst = tris[doff+ji];
                            for (int k=0;k<3;++k) {
                                float vx=src.v[k][0]*dm.scale, vy=src.v[k][1]*dm.scale, vz=src.v[k][2]*dm.scale;
                                dst.v[k][0] = rot.right.x*vx + rot.up.x*vy + rot.forward.x*vz + tx;
                                dst.v[k][1] = rot.right.y*vx + rot.up.y*vy + rot.forward.y*vz + ty;
                                dst.v[k][2] = rot.right.z*vx + rot.up.z*vy + rot.forward.z*vz + tz;
                            }
                        }
                        doff += dyn_mesh_tris[mi].size();
                    }
                    // Robot orbit positions
                    float r_orb = orbit_pairs_test->orbit_radius;
                    orbit_pairs_angle += capture_dt * orbit_pairs_test->orbit_speed;
                    for (size_t pi = 0; pi < orbit_pairs_test->pairs.size(); ++pi) {
                        const auto& ref = orbit_pairs_test->pairs[pi];
                        auto& robot = cfg.robots[ref.robot_idx];
                        const auto& dm = cfg.dynamic_meshes[ref.mesh_idx];
                        float cx=dm.w_pose.pos.x, cy=dm.w_pose.pos.y, cz=dm.w_pose.pos.z;
                        float angle = orbit_pairs_angle + float(pi)*3.14159f;
                        float x = cx + r_orb*std::cos(angle);
                        float y = cy;
                        float z = cz + r_orb*std::sin(angle);
                        robot.w_pose.pos = {x,y,z};
                        float dxv=cx-x, dyv=cy-y, dzv=cz-z;
                        float len = std::sqrt(dxv*dxv+dyv*dyv+dzv*dzv);
                        if (len > 1e-6f) robot.w_pose.forward = {dxv/len, dyv/len, dzv/len};
                        robot.w_pose.up = {0,1,0};
                        robot.w_pose.right = {
                            robot.w_pose.up.y*robot.w_pose.forward.z - robot.w_pose.up.z*robot.w_pose.forward.y,
                            robot.w_pose.up.z*robot.w_pose.forward.x - robot.w_pose.up.x*robot.w_pose.forward.z,
                            robot.w_pose.up.x*robot.w_pose.forward.y - robot.w_pose.up.y*robot.w_pose.forward.x
                        };
                        for (auto& lidar : robot.lidars) computeLidarWorldPose(robot.w_pose, lidar);
                    }
                } else if (lidar_orbit_test && active_test == "lidar_orbit_test") {
                    size_t doff = static_tris.size();
                    for (size_t i = 0; i < cfg.dynamic_meshes.size(); ++i) {
                        const auto& dm = cfg.dynamic_meshes[i];
                        for (size_t ji = 0; ji < dyn_mesh_tris[i].size(); ++ji) {
                            const auto& src = dyn_mesh_tris[i][ji];
                            auto& dst = tris[doff+ji];
                            for (int k=0;k<3;++k) {
                                float vx=src.v[k][0]*dm.scale, vy=src.v[k][1]*dm.scale, vz=src.v[k][2]*dm.scale;
                                dst.v[k][0] = dm.w_pose.right.x*vx + dm.w_pose.up.x*vy + dm.w_pose.forward.x*vz + dm.w_pose.pos.x;
                                dst.v[k][1] = dm.w_pose.right.y*vx + dm.w_pose.up.y*vy + dm.w_pose.forward.y*vz + dm.w_pose.pos.y;
                                dst.v[k][2] = dm.w_pose.right.z*vx + dm.w_pose.up.z*vy + dm.w_pose.forward.z*vz + dm.w_pose.pos.z;
                            }
                        }
                        doff += dyn_mesh_tris[i].size();
                    }
                } else {
                    for (size_t i = 0; i < cfg.dynamic_meshes.size(); ++i) {
                        auto& pose = cfg.dynamic_meshes[i].w_pose;
                        const auto& dm = cfg.dynamic_meshes[i];
                        float& tx = pose.pos.x, &ty = pose.pos.y, &tz = pose.pos.z;
                        float teuler[3] = {}; float cur_scale[3] = {dm.scale, dm.scale, dm.scale};
                        bool dyn_rand_ori = false;
                        if (has_world_aabb && dynmesh_mask[i]) {
                            int pf = cfg.random_dynamic_pos_interpolate_frames;
                            int rf = cfg.random_dynamic_rot_interpolate_frames;
                            int sf = cfg.random_dynamic_scale_interpolate_frames;
                            bool pe = pf>=0, re = rf>=0, se = sf>=0;
                            if (pe || re || se) {
                                int tot = 0;
                                if (pe) tot=std::max(tot,pf);
                                if (re) tot=std::max(tot,rf);
                                if (se) tot=std::max(tot,sf);
                                auto& ts = dynmesh_ts[i];
                                if (!ts.initialized || ts.interp_frame >= tot) {
                                    if (!ts.initialized) {
                                        ts.prev[0]=dm.w_pose.pos.x; ts.prev[1]=dm.w_pose.pos.y; ts.prev[2]=dm.w_pose.pos.z;
                                        ts.prev_euler[0]=ts.prev_euler[1]=ts.prev_euler[2]=0.f;
                                        for (int e=0;e<3;++e) ts.prev_scale[e]=dm.scale;
                                    } else {
                                        for (int e=0;e<3;++e) ts.prev[e]=ts.next[e];
                                        for (int e=0;e<3;++e) ts.prev_euler[e]=ts.next_euler[e];
                                        for (int e=0;e<3;++e) ts.prev_scale[e]=ts.next_scale[e];
                                    }
                                    if (pe) { auto d=pick_dest(); ts.next[0]=d[0]; ts.next[1]=d[1]; ts.next[2]=d[2]; }
                                    else    for (int e=0;e<3;++e) ts.next[e]=ts.prev[e];
                                    if (re) for (int e=0;e<3;++e) ts.next_euler[e]=pick_angle();
                                    else    for (int e=0;e<3;++e) ts.next_euler[e]=ts.prev_euler[e];
                                    if (se) pick_scale3(ts.next_scale);
                                    else for (int e=0;e<3;++e) ts.next_scale[e]=ts.prev_scale[e];
                                    ts.interp_frame=0; ts.initialized=true;
                                }
                                if (pe) {
                                    float pa = (pf==0)?1.f:std::min(1.f,float(ts.interp_frame)/float(pf));
                                    tx=ts.prev[0]*(1.f-pa)+ts.next[0]*pa;
                                    ty=ts.prev[1]*(1.f-pa)+ts.next[1]*pa;
                                    tz=ts.prev[2]*(1.f-pa)+ts.next[2]*pa;
                                }
                                if (re) {
                                    float ra=(rf==0)?1.f:std::min(1.f,float(ts.interp_frame)/float(rf));
                                    for (int e=0;e<3;++e) teuler[e]=ts.prev_euler[e]*(1.f-ra)+ts.next_euler[e]*ra;
                                    dyn_rand_ori=true;
                                }
                                if (se) {
                                    float sa=(sf==0)?1.f:std::min(1.f,float(ts.interp_frame)/float(sf));
                                    for (int e=0;e<3;++e) cur_scale[e]=ts.prev_scale[e]*(1.f-sa)+ts.next_scale[e]*sa;
                                }
                                ts.interp_frame++;
                            }
                        }
                        if (dyn_rand_ori) rotationFromEuler(teuler[0],teuler[1],teuler[2],pose);
                        else {
                            pose.right.x=pose.up.y*pose.forward.z-pose.up.z*pose.forward.y;
                            pose.right.y=pose.up.z*pose.forward.x-pose.up.x*pose.forward.z;
                            pose.right.z=pose.up.x*pose.forward.y-pose.up.y*pose.forward.x;
                        }
                        for (size_t ji = 0; ji < dyn_mesh_tris[i].size(); ++ji) {
                            const auto& src = dyn_mesh_tris[i][ji];
                            auto& dst = tris[dyn_write+ji];
                            for (int k=0;k<3;++k) {
                                float vx=src.v[k][0]*cur_scale[0], vy=src.v[k][1]*cur_scale[1], vz=src.v[k][2]*cur_scale[2];
                                dst.v[k][0]=pose.right.x*vx+pose.up.x*vy+pose.forward.x*vz+tx;
                                dst.v[k][1]=pose.right.y*vx+pose.up.y*vy+pose.forward.y*vz+ty;
                                dst.v[k][2]=pose.right.z*vx+pose.up.z*vy+pose.forward.z*vz+tz;
                            }
                        }
                        dyn_write += dyn_mesh_tris[i].size();
                    }
                }

                // ------------------------------------------------------------------
                // Robot transform
                // ------------------------------------------------------------------
                if (lidar_orbit_test) {
                    if (lidar_orbit_test->dynamic_mesh_index >= 0 &&
                        lidar_orbit_test->dynamic_mesh_index < (int)cfg.dynamic_meshes.size()) {
                        const auto& dm = cfg.dynamic_meshes[lidar_orbit_test->dynamic_mesh_index];
                        float cx=dm.w_pose.pos.x, cy=dm.w_pose.pos.y, cz=dm.w_pose.pos.z;
                        float r_orb = lidar_orbit_test->orbit_radius;
                        orbit_angle += capture_dt * lidar_orbit_test->orbit_speed;
                        for (size_t ci = 0; ci < lidar_orbit_test->lidar_candidates.size(); ++ci) {
                            const auto& ref = lidar_orbit_test->lidar_candidates[ci];
                            auto& robot = cfg.robots[ref.robot_idx];
                            float angle = orbit_angle + float(ci)*3.14159f;
                            float x=cx+r_orb*std::cos(angle), y=cy, z=cz+r_orb*std::sin(angle);
                            robot.w_pose.pos = {x,y,z};
                            float dxv=cx-x, dyv=cy-y, dzv=cz-z;
                            float len=std::sqrt(dxv*dxv+dyv*dyv+dzv*dzv);
                            if (len>1e-6f) robot.w_pose.forward = {dxv/len,dyv/len,dzv/len};
                            robot.w_pose.up = {0,1,0};
                            robot.w_pose.right = {
                                robot.w_pose.up.y*robot.w_pose.forward.z - robot.w_pose.up.z*robot.w_pose.forward.y,
                                robot.w_pose.up.z*robot.w_pose.forward.x - robot.w_pose.up.x*robot.w_pose.forward.z,
                                robot.w_pose.up.x*robot.w_pose.forward.y - robot.w_pose.up.y*robot.w_pose.forward.x
                            };
                            computeLidarWorldPose(robot.w_pose, robot.lidars[ref.lidar_idx]);
                        }
                    }
                    for (size_t ri = 0; ri < cfg.robots.size(); ++ri) {
                        auto& robot = cfg.robots[ri];
                        robot.w_pose.right = {
                            robot.w_pose.up.y*robot.w_pose.forward.z - robot.w_pose.up.z*robot.w_pose.forward.y,
                            robot.w_pose.up.z*robot.w_pose.forward.x - robot.w_pose.up.x*robot.w_pose.forward.z,
                            robot.w_pose.up.x*robot.w_pose.forward.y - robot.w_pose.up.y*robot.w_pose.forward.x
                        };
                        for (auto& lidar : robot.lidars) computeLidarWorldPose(robot.w_pose, lidar);
                    }
                } else {
                    for (size_t ri = 0; ri < cfg.robots.size(); ++ri) {
                        auto& robot = cfg.robots[ri];
                        auto& pose  = robot.w_pose;
                        // When robots_share_pose, copy robot[0]'s already-computed pose
                        // and recompute each robot's own lidar world poses from it.
                        if (cfg.random_robots_share_pose && ri > 0) {
                            pose = cfg.robots[0].w_pose;
                            for (size_t li = 0; li < robot.lidars.size(); ++li)
                                if (robot_lidar_mask[ri][li]) computeLidarWorldPose(pose, robot.lidars[li]);
                            continue;
                        }
                        float& rpx=pose.pos.x, &rpy=pose.pos.y, &rpz=pose.pos.z;
                        float reuler[3] = {}; bool rand_ori = false;
                        if (has_world_aabb && robot_mask[ri]) {
                            int pf = cfg.random_dynamic_pos_interpolate_frames;
                            int rf = cfg.random_dynamic_rot_interpolate_frames;
                            bool pe=pf>=0, re=rf>=0;
                            if (pe || re) {
                                int tot = pe&&re ? std::max(pf,rf) : pe ? pf : rf;
                                auto& ts = robot_ts[ri];
                                if (!ts.initialized || ts.interp_frame >= tot) {
                                    if (!ts.initialized) {
                                        ts.prev[0]=robot.w_pose.pos.x; ts.prev[1]=robot.w_pose.pos.y; ts.prev[2]=robot.w_pose.pos.z;
                                        ts.prev_euler[0]=ts.prev_euler[1]=ts.prev_euler[2]=0.f;
                                    } else {
                                        for (int e=0;e<3;++e) ts.prev[e]=ts.next[e];
                                        for (int e=0;e<3;++e) ts.prev_euler[e]=ts.next_euler[e];
                                    }
                                    if (pe) { auto d=pick_dest(); ts.next[0]=d[0]; ts.next[1]=d[1]; ts.next[2]=d[2]; }
                                    else    for (int e=0;e<3;++e) ts.next[e]=ts.prev[e];
                                    if (re) for (int e=0;e<3;++e) ts.next_euler[e]=pick_angle();
                                    else    for (int e=0;e<3;++e) ts.next_euler[e]=ts.prev_euler[e];
                                    ts.interp_frame=0; ts.initialized=true;
                                }
                                if (pe) {
                                    float pa=(pf==0)?1.f:std::min(1.f,float(ts.interp_frame)/float(pf));
                                    rpx=ts.prev[0]*(1.f-pa)+ts.next[0]*pa;
                                    rpy=ts.prev[1]*(1.f-pa)+ts.next[1]*pa;
                                    rpz=ts.prev[2]*(1.f-pa)+ts.next[2]*pa;
                                }
                                if (re) {
                                    float ra=(rf==0)?1.f:std::min(1.f,float(ts.interp_frame)/float(rf));
                                    for (int e=0;e<3;++e) reuler[e]=ts.prev_euler[e]*(1.f-ra)+ts.next_euler[e]*ra;
                                    rand_ori=true;
                                }
                                ts.interp_frame++;
                            }
                        }
                        if (rand_ori) rotationFromEuler(reuler[0],reuler[1],reuler[2],pose);
                        else {
                            pose.right.x=pose.up.y*pose.forward.z-pose.up.z*pose.forward.y;
                            pose.right.y=pose.up.z*pose.forward.x-pose.up.x*pose.forward.z;
                            pose.right.z=pose.up.x*pose.forward.y-pose.up.y*pose.forward.x;
                        }
                        for (size_t li = 0; li < robot.lidars.size(); ++li)
                            if (robot_lidar_mask[ri][li]) computeLidarWorldPose(pose, robot.lidars[li]);
                    }
                }

                // ------------------------------------------------------------------
                // Backend dispatch
                // ------------------------------------------------------------------
                FrameTimings t = {};
                std::vector<GrcaHit> hits;
                auto tSim = clk::now();

                if      (be == "grca-cpu") run_cpu_frame(cpuState, cfg, tris, hits, t);
#ifdef HAVE_CUDA
                else if (be == "grca-cuda") run_cuda_frame(cudaState, cfg, tris, hits, t);
#endif
#ifdef HAVE_OPTIX
                else if (be == "optix" || be == "optix-crti") run_optix_frame(optixState, cfg, tris, hits, t);
#endif
#ifdef HAVE_EMBREE
                else if (be == "embree")   run_embree_frame(embreeState, cfg, tris, hits, t);
#endif
#ifdef HAVE_TINYBVH
                else if (be == "tinybvh-cpu") run_tinybvh_cpu_frame(tinybvhCpuState, cfg, tris, hits, t);
                else if (be == "tinybvh-gpu") run_tinybvh_gpu_frame(tinybvhGpuState, cfg, tris, hits, t);
#endif
#ifdef HAVE_EMBREE
                else if (be == "hybrid-cpu") run_hybrid_cpu_frame(hybridCpuState, cfg, tris, hits, t);
#endif
#if defined(HAVE_OPTIX) && defined(HAVE_CUDA)
                else if (be == "hybrid-gpu") run_hybrid_gpu_frame(hybridGpuState, cfg, tris, hits, t);
#endif
                else { std::cerr << "Unknown backend: " << be << "\n"; break; }

                t.ms_lidar_sim_cpu = elapsed_ms(tSim, clk::now());
                if (t.num_bat >= 0) {
                    if (t.num_bat < bat_min) bat_min = t.num_bat;
                    if (t.num_bat > bat_max) bat_max = t.num_bat;
                    bat_sum += t.num_bat;
                    ++bat_frame_count;
                    if (t.max_bat_capacity > 0) bat_capacity = t.max_bat_capacity;
                }
                if (t.num_early_t >= 0) {
                    if (t.num_early_t < early_t_min) early_t_min = t.num_early_t;
                    if (t.num_early_t > early_t_max) early_t_max = t.num_early_t;
                    early_t_sum += t.num_early_t;
                }
                if (t.num_rtic >= 0) {
                    if (t.num_rtic < rtic_min) rtic_min = t.num_rtic;
                    if (t.num_rtic > rtic_max) rtic_max = t.num_rtic;
                    rtic_sum += t.num_rtic;
                }
                t.ms_total_cpu     = elapsed_ms(tStart, clk::now());

                // FPS / real-time factor
                double now_rt = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                double frame_real_dt = (prev_frame_real_time > 0.0)
                    ? now_rt - prev_frame_real_time
                    : t.ms_total_cpu / 1000.0;
                prev_frame_real_time = now_rt;
                t.sim_frame_rate   = (frame_real_dt > 0.0) ? 1.0 / frame_real_dt : 0.0;
                t.real_time_factor = (frame_real_dt > 0.0) ? capture_dt / frame_real_dt : 0.0;

                fprintf(stderr, "[frame %d  t=%.3fs]\n", frame, sim_time);
                printTimings(runIdx, batchIdx, cfg, totalRays, static_tris.size(), tris.size() - static_tris.size(), t, session_label);
                latest_hits = hits;
                last_vis_frame = frame;
                last_vis_sim_time = sim_time;

                // Accumulate for CSV summary
                if (perf_record_frames > -1 && frame < perf_record_frames) {
                    sum_fps          += t.sim_frame_rate;
                    sum_rtf          += t.real_time_factor;
                    sum_lidar_sim    += t.ms_lidar_sim_cpu;
                    sum_bvh_rebuild_gpu += t.ms_bvh_rebuild_gpu;
                    sum_bvh_refit_gpu   += t.ms_bvh_refit_gpu;
                    sum_bvh_rebuild_cpu += t.ms_bvh_rebuild_cpu;
                    sum_bvh_refit_cpu   += t.ms_bvh_refit_cpu;
                    sum_trace        += t.ms_trace_gpu;
                    sum_trace_cpu    += t.ms_trace_cpu;
                    sum_sync_rb      += t.ms_sync_readback;
                    sum_dyn_up       += t.ms_dynamic_upload;
                    sum_fillhits     += t.ms_fillhits;
                    sum_lidar_sim_gpu+= t.ms_lidar_sim_gpu;
                    sum_total_cpu    += t.ms_total_cpu;
                    sum_rb           += t.ms_readback_cpu;
                    if (t.ms_lidar_sim_cpu > 0) {
                        min_lidar_sim_cpu = std::min(min_lidar_sim_cpu, t.ms_lidar_sim_cpu);
                        max_lidar_sim_cpu = std::max(max_lidar_sim_cpu, t.ms_lidar_sim_cpu);
                    }
                    if (t.ms_lidar_sim_gpu > 0) {
                        min_lidar_sim_gpu = std::min(min_lidar_sim_gpu, t.ms_lidar_sim_gpu);
                        max_lidar_sim_gpu = std::max(max_lidar_sim_gpu, t.ms_lidar_sim_gpu);
                    }
                    perf_frame_count++;
                }

                // Visualizer update
                if (cfg.visualizer_on && vis) {
                    if (!cfg.dynamic_meshes.empty()) {
                        std::vector<VisMesh> dvm;
                        size_t doff = static_tris.size();
                        for (size_t i = 0; i < cfg.dynamic_meshes.size(); ++i) {
                            const auto& dm = cfg.dynamic_meshes[i];
                            VisMesh vm;
                            vm.tris       = tris.data() + doff;
                            vm.count      = dyn_mesh_tris[i].size();
                            vm.color[0]   = dm.color[0]; vm.color[1] = dm.color[1]; vm.color[2] = dm.color[2];
                            vm.tri_colors = (cfg.mesh_vis_on >= 3 || dm.color_override || dyn_mesh_colors[i].empty()) ? nullptr : dyn_mesh_colors[i].data();
                            dvm.push_back(vm);
                            doff += dyn_mesh_tris[i].size();
                        }
                        vis_update_dynamic_meshes(vis, dvm);
                        // Recompute world-space AABBs for dynamic meshes
                        const size_t nstatic = cfg.static_meshes.size() + (show_fallback_static_aabb ? 1 : 0);
                        vis_aabbs.resize(nstatic + cfg.dynamic_meshes.size());
                        doff = static_tris.size();
                        for (size_t i = 0; i < cfg.dynamic_meshes.size(); ++i) {
                            VisAABB& va = vis_aabbs[nstatic + i];
                            float xmn = std::numeric_limits<float>::max(),    ymn = xmn, zmn = xmn;
                            float xmx = std::numeric_limits<float>::lowest(), ymx = xmx, zmx = xmx;
                            const Tri3* base = tris.data() + doff;
                            const size_t nc = dyn_mesh_tris[i].size();
                            for (size_t ji = 0; ji < nc; ++ji)
                                for (int k = 0; k < 3; ++k) {
                                    xmn=std::min(xmn,base[ji].v[k][0]); xmx=std::max(xmx,base[ji].v[k][0]);
                                    ymn=std::min(ymn,base[ji].v[k][1]); ymx=std::max(ymx,base[ji].v[k][1]);
                                    zmn=std::min(zmn,base[ji].v[k][2]); zmx=std::max(zmx,base[ji].v[k][2]);
                                }
                            va.aabb[0]=xmn; va.aabb[1]=ymn; va.aabb[2]=zmn;
                            va.aabb[3]=xmx; va.aabb[4]=ymx; va.aabb[5]=zmx;
                            doff += nc;
                        }
                        vis_update_aabbs(vis, vis_aabbs);
                    }
                    // Update lidar cylinders + cones (robots may have moved)
                    {
                        std::vector<VisLidar> vlidars;
                        for (const auto& robot : cfg.robots)
                            for (const auto& lidar : robot.lidars) {
                                VisLidar vl;
                                vl.pos[0]    = lidar.w_pose.pos.x;
                                vl.pos[1]    = lidar.w_pose.pos.y;
                                vl.pos[2]    = lidar.w_pose.pos.z;
                                vl.fwd[0]    = lidar.w_pose.up.x;
                                vl.fwd[1]    = lidar.w_pose.up.y;
                                vl.fwd[2]    = lidar.w_pose.up.z;
                                vl.up[0]     = lidar.w_pose.forward.x;
                                vl.up[1]     = lidar.w_pose.forward.y;
                                vl.up[2]     = lidar.w_pose.forward.z;
                                vl.vmin      = lidar.vmin;
                                vl.vmax      = lidar.vmax;
                                vl.vnum      = lidar.vnum;
                                vl.range_max = lidar.range_max;
                                vlidars.push_back(vl);
                            }
                        vis_set_lidars(vis, vlidars);
                        vis_set_lidar_cones(vis, vlidars);
                    }
                    vis_update(vis, hits, cfg, frame, sim_time);
                    if (!vis_running(vis)) running = false;
                }

                // Hit-view CSV (manual 'o' press or auto-capture)
                bool doCapture = (auto_hit_view_global >= 0 && frame == auto_hit_view_global);
                doCapture = doCapture || (vis && vis_capture_requested(vis));
                if (doCapture) {
                    std::vector<int> ro_ids, li_ids;
                    for (size_t ri = 0; ri < cfg.robots.size(); ++ri)
                        for (size_t li = 0; li < cfg.robots[ri].lidars.size(); ++li) {
                            size_t nRays = (size_t)cfg.robots[ri].lidars[li].hnum * cfg.robots[ri].lidars[li].vnum;
                            for (size_t k = 0; k < nRays; ++k) { ro_ids.push_back((int)ri); li_ids.push_back((int)li); }
                        }
                    std::string csvPath = make_timestamped_csv_name("hit_view_" + be, cfg, batch_dir);
                    writeCSV(csvPath, hits, ro_ids, li_ids);
                    fprintf(stderr, "  Wrote %s  (%zu rays)\n", csvPath.c_str(), hits.size());
                }

                // Perf CSV row
                if (perf_record_frames > -1 && frame < perf_record_frames) {
                    write_perf_csv_row(perf_csv_file, frame, t);
                    perf_csv_file.flush();
                }

                // Hit-distance CSV — record randomized 10% sample (fixed per run)
                if (hit_dist_record_frames > -1 && frame < hit_dist_record_frames &&
                    hit_dist_sample_frames.count(frame)) {
                    if (!hit_dist_csv_out) {
                        hit_dist_csv_name = make_timestamped_csv_name("hit_dist", cfg, batch_dir);
                        hit_dist_csv_out  = std::make_unique<std::ofstream>(hit_dist_csv_name);
                        *hit_dist_csv_out << "frame,dist\n";
                    }
                    for (const auto& h : hits)
                        *hit_dist_csv_out << frame << "," << h.dist << "\n";
                }

                ++frame;
                sim_time += capture_dt;
                { double r = get_process_ram_mb(); if (r > batch_max_ram_mb) batch_max_ram_mb = r; }
#ifdef HAVE_CUDA
                if ((be == "grca-cuda" || be == "optix" || be == "optix-crti" || be == "hybrid-gpu") && vram_total_gpu > 0) {
                    size_t curFree2 = 0, curTotal2 = 0;
                    cudaMemGetInfo(&curFree2, &curTotal2);
                    size_t used = curTotal2 - curFree2;
                    if (used > batch_max_vram) batch_max_vram = used;
                    double deltaMB = (double)(used > vram_main_start ? used - vram_main_start : 0) / (1024.0 * 1024.0);
                    double meshMB  = (double)(init_tris.size() * sizeof(Tri3)) / (1024.0 * 1024.0);
                    fprintf(stderr, "[VRAM] frame %d: +%.1f MB (mesh: %.1f MB, other: %.1f MB | total: %.1f MB)\n",
                            frame - 1,
                            deltaMB, meshMB, deltaMB - meshMB,
                            (double)used / (1024.0 * 1024.0));
                }
#endif
            } // end frame loop

            // ------------------------------------------------------------------
            // Write runtime info txt
            // ------------------------------------------------------------------
            {
                bool has_bat = bat_max >= 0;
#ifdef HAVE_CUDA
                bool has_vram = (be == "grca-cuda" || be == "optix" || be == "hybrid-gpu") && batch_max_vram > 0;
#else
                bool has_vram = false;
#endif
                bool has_ram = batch_max_ram_mb > 0.0;
                if (has_bat || has_vram || has_ram) {
                    std::string info_txt = batch_dir + "/runtime_info_" + be + ".txt";
                    FILE* rf = fopen(info_txt.c_str(), "w");
                    if (rf) {
                        fprintf(rf, "backend:    %s\n\n", be.c_str());
                        if (be == "grca-cuda" || be == "grca" || be == "hybrid-gpu") {
                            fprintf(rf, "sat_max_sweep_diff:   %d\n", cfg.cuda_sat_max_sweep_diff);
                            fprintf(rf, "sat_max_channel_diff: %d\n\n", cfg.cuda_sat_max_channel_diff);
                        } else if (be == "grca-cpu" || be == "hybrid-cpu") {
                            fprintf(rf, "sat_max_sweep_diff:   %d\n", cfg.cpu_sat_max_sweep_diff);
                            fprintf(rf, "sat_max_channel_diff: %d\n\n", cfg.cpu_sat_max_channel_diff);
                        }
                        if (has_bat) {
                            int total_tris = (int)init_tris.size();
                            double d   = total_tris ? total_tris : 1;
                            double fc  = bat_frame_count ? bat_frame_count : 1;
                            double bat_avg  = bat_sum / fc;
                            double et_avg   = early_t_max >= 0 ? (double)early_t_sum / fc : -1.0;
                            double rtic_avg = rtic_max  >= 0 ? (double)rtic_sum  / fc : -1.0;
                            fprintf(rf, "=== Triangle filter stats (debug=true, %d frames) ===\n", bat_frame_count);
                            fprintf(rf, "total_tris:    %d\n\n", total_tris);
                            fprintf(rf, "%-14s %15s  %15s  %15s  %10s\n", "", "min", "avg", "max", "avg%%");
                            fprintf(rf, "%-14s %15d  %15.1f  %15d  %9.2f%%\n", "bat:",
                                    bat_min, bat_avg, bat_max, 100.0 * bat_avg / d);
                            if (bat_capacity > 0) {
                                double cap = (double)bat_capacity;
                                fprintf(rf, "%-14s %15d\n", "bat_capacity:", bat_capacity);
                                fprintf(rf, "%-14s %14.2f%%  %14.2f%%  %14.2f%%\n", "bat_util:",
                                        100.0 * bat_min / cap,
                                        100.0 * bat_avg / cap,
                                        100.0 * bat_max / cap);
                            }
                            if (early_t_max >= 0) {
                                fprintf(rf, "%-14s %15d  %15.1f  %15d\n", "early_t:",
                                        early_t_min, et_avg, early_t_max);
                                double sat_avg = et_avg - bat_avg;
                                fprintf(rf, "%-14s %15s  %15.1f  %15s  (approx = early_t - bat)\n",
                                        "sat:", "--", sat_avg < 0 ? 0.0 : sat_avg, "--");
                            }
                            if (rtic_max >= 0)
                                fprintf(rf, "%-14s %15d  %15.1f  %15d\n", "rtic:",
                                        rtic_min, rtic_avg, rtic_max);
                        }
#ifdef HAVE_CUDA
                        if (has_vram) {
                            double maxMB   = (double)batch_max_vram / (1024.0 * 1024.0);
                            double deltaMB = (double)(batch_max_vram > vram_main_start
                                                      ? batch_max_vram - vram_main_start : 0)
                                             / (1024.0 * 1024.0);
                            double meshMB  = (double)(init_tris.size() * sizeof(Tri3)) / (1024.0 * 1024.0);
                            if (has_bat) fprintf(rf, "\n");
                            fprintf(rf, "=== VRAM (peak this batch) ===\n");
                            fprintf(rf, "vram_delta: +%.1f MB\n", deltaMB);
                            fprintf(rf, "vram_mesh:  %.1f MB\n",  meshMB);
                            fprintf(rf, "vram_other: %.1f MB\n",  deltaMB - meshMB);
                            fprintf(rf, "vram_total: %.1f MB\n",  maxMB);
                        }
#endif
                        if (has_ram) {
                            if (has_bat || has_vram) fprintf(rf, "\n");
                            fprintf(rf, "=== RAM (peak this batch) ===\n");
                            fprintf(rf, "ram_peak:   %.1f MB\n", batch_max_ram_mb);
                        }
                        fclose(rf);
                        fprintf(stderr, "Wrote %s\n", info_txt.c_str());
                    }
                }
            }

            // ------------------------------------------------------------------
            // Write performance CSV
            // ------------------------------------------------------------------
            if (perf_record_frames > -1 && perf_frame_count > 0) {
                auto avrow = [&](const char* label, double val) {
                    perf_csv_file << label << "," << val << "\n";
                };
                perf_csv_file << "# Init CPU (ms):," << perf_init_cpu_ms << "\n";
                avrow("# Average FPS:",                 sum_fps              / perf_frame_count);
                avrow("# Average Real-Time Factor:",    sum_rtf              / perf_frame_count);
                avrow("# Average Lidar Sim CPU (ms):",  sum_lidar_sim        / perf_frame_count);
                avrow("# Average Lidar Sim GPU (ms):",  sum_lidar_sim_gpu    / perf_frame_count);
                avrow("# Average Total CPU (ms):",      sum_total_cpu        / perf_frame_count);
                avrow("# Average Dynamic Upload (ms):", sum_dyn_up           / perf_frame_count);
                avrow("# Average BVH Rebuild GPU (ms):",sum_bvh_rebuild_gpu  / perf_frame_count);
                avrow("# Average BVH Refit GPU (ms):",  sum_bvh_refit_gpu    / perf_frame_count);
                avrow("# Average BVH Rebuild CPU (ms):",sum_bvh_rebuild_cpu  / perf_frame_count);
                avrow("# Average BVH Refit CPU (ms):",  sum_bvh_refit_cpu    / perf_frame_count);
                avrow("# Average Trace GPU (ms):",      sum_trace            / perf_frame_count);
                avrow("# Average Trace CPU (ms):",      sum_trace_cpu        / perf_frame_count);
                avrow("# Average Readback (ms):",       sum_rb               / perf_frame_count);
                avrow("# Average Sync Readback (ms):",  sum_sync_rb          / perf_frame_count);
                avrow("# Average Fill Hits (ms):",      sum_fillhits         / perf_frame_count);
                fprintf(stderr, "Wrote %s\n", perf_csv_name.c_str());
            }

            // ------------------------------------------------------------------
            // Store per-backend result for batch summary
            // ------------------------------------------------------------------
            {
                BatchBackendResult br;
                br.backend      = be;
                br.label        = backendLabel(cfg);
                br.api_label    = backendApiLabel(cfg);
                br.hit_dist_csv = hit_dist_csv_name;
                br.pose_csv     = pose_csv_name;
                br.perf_frames  = perf_frame_count;
                br.batch_lsr   = batch_lsr;
                br.init_cpu_ms = perf_init_cpu_ms;
                if (perf_frame_count > 0) {
                    br.avg_fps               = sum_fps              / perf_frame_count;
                    br.avg_rtf               = sum_rtf              / perf_frame_count;
                    br.avg_total_cpu_ms      = sum_total_cpu        / perf_frame_count;
                    br.avg_lidar_sim_cpu     = sum_lidar_sim        / perf_frame_count;
                    br.min_lidar_sim_cpu     = (min_lidar_sim_cpu < 1e29) ? min_lidar_sim_cpu : 0;
                    br.max_lidar_sim_cpu     = max_lidar_sim_cpu;
                    br.avg_lidar_sim_gpu     = sum_lidar_sim_gpu    / perf_frame_count;
                    br.min_lidar_sim_gpu     = (min_lidar_sim_gpu < 1e29) ? min_lidar_sim_gpu : 0;
                    br.max_lidar_sim_gpu     = max_lidar_sim_gpu;
                    br.avg_trace_gpu         = sum_trace            / perf_frame_count;
                    br.avg_trace_cpu         = sum_trace_cpu        / perf_frame_count;
                    br.avg_dynamic_upload_ms = sum_dyn_up           / perf_frame_count;
                    br.avg_bvh_rebuild_gpu_ms= sum_bvh_rebuild_gpu  / perf_frame_count;
                    br.avg_bvh_refit_gpu_ms  = sum_bvh_refit_gpu    / perf_frame_count;
                    br.avg_bvh_rebuild_cpu_ms= sum_bvh_rebuild_cpu  / perf_frame_count;
                    br.avg_bvh_refit_cpu_ms  = sum_bvh_refit_cpu    / perf_frame_count;
                    br.avg_readback_ms       = sum_rb               / perf_frame_count;
                    br.avg_sync_readback_ms  = sum_sync_rb          / perf_frame_count;
                    br.avg_fill_hits_ms      = sum_fillhits         / perf_frame_count;
                }
                batch_results.push_back(br);
            }
#ifdef HAVE_CUDA
            if ((be == "grca-cuda" || be == "optix" || be == "optix-crti") && batch_max_vram > 0) {
                std::string label = "run" + std::to_string(runIdx)
                                  + "/batch" + std::to_string(batchIdx)
                                  + "/" + be;
                all_batch_max_vram.push_back({label, batch_max_vram});
            }
#endif

            // ------------------------------------------------------------------
            // Backend cleanup
            // ------------------------------------------------------------------
            if (cpuState)    { destroy_cpu(cpuState);    cpuState    = nullptr; }
#ifdef HAVE_CUDA
            if (cudaState)   { destroy_cuda(cudaState);  cudaState   = nullptr; }
#endif
#ifdef HAVE_OPTIX
            if (optixState)  { destroy_optix(optixState); optixState = nullptr; }
#endif
#ifdef HAVE_EMBREE
            if (embreeState) { destroy_embree(embreeState); embreeState = nullptr; }
#endif
#ifdef HAVE_TINYBVH
            if (tinybvhCpuState) { destroy_tinybvh_cpu(tinybvhCpuState); tinybvhCpuState = nullptr; }
            if (tinybvhGpuState) { destroy_tinybvh_gpu(tinybvhGpuState); tinybvhGpuState = nullptr; }
#endif
#ifdef HAVE_EMBREE
            if (hybridCpuState) { destroy_hybrid_cpu(hybridCpuState); hybridCpuState = nullptr; }
#endif
#if defined(HAVE_OPTIX) && defined(HAVE_CUDA)
            if (hybridGpuState) { destroy_hybrid_gpu(hybridGpuState); hybridGpuState = nullptr; }
#endif
            // Do not destroy visualizer per batch; destroy after all batches

        } // end backend loop

        // Only write summary and results CSV if at least one backend succeeded
        bool has_valid_result = false;
        for (const auto& r : batch_results) {
            if (r.perf_frames > 0) { has_valid_result = true; break; }
        }
        if (has_valid_result) {
            // Write per-batch CSV to session_dir/run_N/batch_M/
            std::string run_dir = session_dir + "/run_" + std::to_string(runIdx);
            std::string batch_dir = run_dir + "/batch_" + std::to_string(batchIdx);
            std::filesystem::create_directories(batch_dir);
            std::string results_csv_path = batch_dir
                + "/results_run" + std::to_string(runIdx)
                + "_batch"       + std::to_string(batchIdx)
                + ".csv";
            print_batch_summary(runIdx, batchIdx, results_csv_path, batch_results);
            if (!skip_plot) run_plot_results(results_csv_path);
        }
        } // end batch loop

        // After each run: generate per-run overview PNG immediately.
        if (!skip_plot) run_plot_results(session_dir, runIdx);

    } // end tests loop

    // After all runs: generate session overview only (per-batch/run PNGs already exist).
    if (!skip_plot) run_plot_results(session_dir, -1, true);

    session_zips.push_back(session_dir);

    ++robot_idx;
    } // end robot loop
    ++dyn_idx;
    } // end dyn_mesh loop
    ++scene_idx;
    } // end scene loop

    if (vis)
        vis_destroy(vis);

    // Zip all session folders into one benchmark-level archive
    if (!session_zips.empty()) {
        // session_zips holds folder paths like "benchmark/YYYYMMDD-user-host"
        // cd into "benchmark/" and zip all session folder names from there
        auto slash = session_zips[0].find_last_of('/');
        std::string bench_parent = (slash != std::string::npos) ? session_zips[0].substr(0, slash) : ".";
        std::ostringstream cmd;
        cmd << "cd \"" << bench_parent << "\" && zip -1 -r \""
            << std::filesystem::path(bench_archive).filename().string() << "\"";
        for (const auto& s : session_zips)
            cmd << " \"" << std::filesystem::path(s).filename().string() << "\"";
        cmd << " 2>&1";
        fprintf(stderr, "\nArchiving all sessions → %s\n", bench_archive.c_str());
        FILE* pipe = popen(cmd.str().c_str(), "r");
        if (pipe) {
            char buf[256];
            while (fgets(buf, sizeof(buf), pipe)) fprintf(stderr, "  %s", buf);
            int rc = pclose(pipe);
            if (rc == 0) fprintf(stderr, "  Done.\n");
            else         fprintf(stderr, "  [zip] exited with code %d\n", rc);
        } else {
            fprintf(stderr, "  [zip] popen failed\n");
        }
    }

#ifdef HAVE_CUDA
    if (!all_batch_max_vram.empty()) {
        fprintf(stderr, "\n=== Max VRAM per batch ===\n");
        for (const auto& kv : all_batch_max_vram) {
            double maxMB   = (double)kv.second / (1024.0 * 1024.0);
            double deltaMB = (double)(kv.second > vram_main_start ? kv.second - vram_main_start : 0) / (1024.0 * 1024.0);
            fprintf(stderr, "  %-40s  +%.1f MB (total: %.1f MB)\n",
                    kv.first.c_str(), deltaMB, maxMB);
        }
    }
#endif

    return 0;
}
