// compare_csv.cpp
// Compares multiple CSV files (with floating-point columns) row-by-row.
// Usage: ./compare_csv [--tol <tolerance>] file1.csv file2.csv [file3.csv ...]
//
// For every pair of files the tool reports per-column statistics:
//   - max absolute difference
//   - mean absolute difference
//   - rows that are within tolerance (matched rows)
//   - match percentage
// A final summary shows which pairs match perfectly (within tolerance).
//
// Performance:
//   - Entire file read in one shot
//   - Manual field splitting — no stringstream, no per-token heap allocation
//   - std::from_chars for number parsing (locale-free, ~5-10x faster than stod)
//   - Column-major storage — comparison inner loop is sequential in memory
//   - string_view into raw buffer for string columns — zero heap allocation per cell
//   - colHasNaN flag eliminates the string-column pre-scan pass
//   - Data portion of each file split into chunks and parsed in parallel threads
//   - str_cols freed for pure numeric columns after loading (saves RAM)
//   - All files loaded in parallel; all pairs compared in parallel

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

using Header = std::vector<std::string>;

// Column-major storage: str_cols[c][r] are string_views into CsvFile::raw
struct CsvFile {
    std::string path;
    std::string raw;                                        // entire file content (owns str_view memory)
    Header      header;
    size_t      numRows = 0;
    std::vector<std::vector<std::string_view>> str_cols;    // str_cols[c][r], views into raw
};

// ---------------------------------------------------------------------------
// Fast number parsing — from_chars, no locale, no exceptions
// ---------------------------------------------------------------------------



// ---------------------------------------------------------------------------
// Chunk parser — parses [p, end) assuming all lines are data rows (no header)
// ---------------------------------------------------------------------------

struct ChunkResult {
    std::vector<std::vector<std::string_view>> str_cols;
    size_t numRows = 0;
};

static ChunkResult parseChunk(const char* p, const char* end, size_t nCols) {
    ChunkResult res;
    res.str_cols.resize(nCols);

    // Estimate row count from average line length to pre-reserve
    size_t dataLen = static_cast<size_t>(end - p);
    // Scan a small sample to estimate average row length
    size_t sampleLen = std::min(dataLen, size_t(4096));
    size_t sampleNewlines = 0;
    for (size_t i = 0; i < sampleLen; ++i)
        if (p[i] == '\n') ++sampleNewlines;
    size_t estRows = sampleNewlines > 0 ? (dataLen / (sampleLen / sampleNewlines)) + 64 : 1024;
    for (size_t c = 0; c < nCols; ++c) {
        res.str_cols[c].reserve(estRows);
    }

    std::vector<std::string_view> tokens;
    tokens.reserve(nCols);



    while (p < end) {
        if (*p == '\n' || *p == '\r') { ++p; continue; }

        tokens.clear();
        while (p < end) {
            const char* fs = p;
            while (p < end && *p != ',' && *p != '\n') ++p;
            size_t len = static_cast<size_t>(p - fs);
            if (len > 0 && fs[len - 1] == '\r') --len;
            tokens.push_back({fs, len});
            if (p >= end || *p == '\n') { if (p < end) ++p; break; }
            ++p; // skip ','
        }
        if (tokens.empty()) continue;

        // Pad short rows; ignore extra columns beyond nCols
        while (tokens.size() < nCols) tokens.push_back({});

        for (size_t c = 0; c < nCols; ++c) {
            res.str_cols[c].push_back(tokens[c]);
        }
        ++res.numRows;
    }


    return res;
}

// ---------------------------------------------------------------------------
// CSV loading — reads entire file, parallel-parses the data section
// ---------------------------------------------------------------------------

// Minimum file data size (bytes) before spinning up extra parse threads.
static constexpr size_t PARALLEL_PARSE_THRESHOLD = 2 * 1024 * 1024; // 2 MB

static CsvFile loadCsv(const std::string& path) {
    CsvFile csv;
    csv.path = path;

    // Read the whole file in one shot
    {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("Cannot open file: " + path);
        std::fseek(fp, 0, SEEK_END);
        long sz = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        if (sz <= 0) { std::fclose(fp); return csv; }
        csv.raw.resize(static_cast<size_t>(sz));
        size_t nread = std::fread(csv.raw.data(), 1, static_cast<size_t>(sz), fp);
        if (nread != static_cast<size_t>(sz))
            csv.raw.resize(nread); // handle short read (e.g. text mode on Windows)
        std::fclose(fp);
    }

    const char* p      = csv.raw.data();
    const char* rawEnd = p + csv.raw.size();

    // ---- No header detection: always treat all rows as data, including the first row ----
    // Generate column names as col0, col1, ...
    // To determine number of columns, scan the first line
    std::vector<std::string_view> tokens;
    tokens.reserve(32);
    const char* lineStart = p;
    while (p < rawEnd && *p != '\n') {
        const char* fs = p;
        while (p < rawEnd && *p != ',' && *p != '\n') ++p;
        size_t len = static_cast<size_t>(p - fs);
        if (len > 0 && fs[len - 1] == '\r') --len;
        tokens.push_back({fs, len});
        if (p >= rawEnd || *p == '\n') break;
        ++p;
    }
    if (p < rawEnd && *p == '\n') ++p;
    for (size_t i = 0; i < tokens.size(); ++i)
        csv.header.push_back("col" + std::to_string(i));
    // Rewind p to start of data
    p = lineStart;

    const size_t nCols     = csv.header.size();
    const char*  dataStart = p;
    const size_t dataSize  = static_cast<size_t>(rawEnd - dataStart);

    // ---- Decide how many parse threads to use --------------------------------
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 2;
    unsigned nThreads = (dataSize >= PARALLEL_PARSE_THRESHOLD)
                        ? std::min(hw, 16u)
                        : 1u;

    // ---- Split data into chunks at newline boundaries -----------------------
    std::vector<std::pair<const char*, const char*>> chunks;
    chunks.reserve(nThreads);
    {
        size_t chunkSize = dataSize / nThreads;
        const char* cp = dataStart;
        for (unsigned i = 0; i < nThreads; ++i) {
            const char* cs = cp;
            if (i == nThreads - 1 || cp >= rawEnd) {
                if (cs < rawEnd) chunks.push_back({cs, rawEnd});
                break;
            }
            const char* ce = cp + chunkSize;
            // Snap to the next newline so we never split mid-row
            while (ce < rawEnd && *ce != '\n') ++ce;
            if (ce < rawEnd) ++ce; // include the '\n'
            chunks.push_back({cs, ce});
            cp = ce;
        }
    }

    // ---- Parse chunks (in parallel if multiple) -----------------------------
    std::vector<ChunkResult> chunkResults;
    chunkResults.reserve(chunks.size());

    if (chunks.size() == 1) {
        chunkResults.push_back(parseChunk(chunks[0].first, chunks[0].second, nCols));
    } else {
        std::vector<std::future<ChunkResult>> futs;
        futs.reserve(chunks.size());
        for (auto [cs, ce] : chunks)
            futs.push_back(std::async(std::launch::async, parseChunk, cs, ce, nCols));
        for (auto& f : futs)
            chunkResults.push_back(f.get());
    }

    // ---- Merge chunk results into csv ---------------------------------------
    size_t totalRows = 0;
    for (auto& cr : chunkResults) totalRows += cr.numRows;

    csv.str_cols.resize(nCols);

    for (size_t c = 0; c < nCols; ++c) {
        csv.str_cols[c].reserve(totalRows);
    }

    // No header skipping: treat all rows as data
    for (auto& cr : chunkResults) {
        for (size_t c = 0; c < nCols; ++c) {
            auto& sc = csv.str_cols[c];
            sc.insert(sc.end(), cr.str_cols[c].begin(), cr.str_cols[c].end());
        }
        csv.numRows += cr.numRows;
    }



    return csv;
}

// ---------------------------------------------------------------------------
// Per-column stats between two files
// ---------------------------------------------------------------------------

struct ColStat {
    double maxAbsDiff    = 0.0;
    double sumAbsDiff    = 0.0;
    size_t withinTolRows = 0;
    size_t totalRows     = 0;

    double meanAbsDiff() const { return totalRows ? sumAbsDiff / totalRows : 0.0; }
    double matchPct()    const { return totalRows ? 100.0 * withinTolRows / totalRows : 0.0; }
};

struct PairResult {
    std::string nameA, nameB;
    std::vector<ColStat> colStats;
    size_t rowsCompared = 0;

    bool allMatch(double minMatchPct) const {
        for (const auto& cs : colStats)
            if (cs.matchPct() < minMatchPct) return false;
        return true;
    }
};

// comparePair is pure (no shared writes) — safe to call concurrently.
static PairResult comparePair(const CsvFile& a, const CsvFile& b, double tol) {
    PairResult result;
    result.nameA = a.path;
    result.nameB = b.path;

    size_t nCols = std::max(a.header.size(), b.header.size());
    result.colStats.resize(nCols);

    size_t nRows = std::max(a.numRows, b.numRows);
    result.rowsCompared = nRows;

    for (size_t c = 0; c < nCols; ++c) {
        auto& cs       = result.colStats[c];
        cs.totalRows   = nRows;
        // Use empty string if column is missing in either file
        const auto& sa = (c < a.str_cols.size()) ? a.str_cols[c] : std::vector<std::string_view>{};
        const auto& sb = (c < b.str_cols.size()) ? b.str_cols[c] : std::vector<std::string_view>{};
        size_t matched = 0;
        double maxDiff = 0.0, sumDiff = 0.0;
        int debug_mismatches = 0;
        auto trim = [](std::string_view s) -> std::string_view {
            size_t start = 0, end = s.size();
            while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
            while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
            return s.substr(start, end - start);
        };
        auto is_numeric = [](std::string_view s) -> bool {
            if (s.empty()) return false;
            char* endptr = nullptr;
            std::string str(s);
            double val = std::strtod(str.c_str(), &endptr);
            return endptr && *endptr == '\0';
        };
        for (size_t r = 0; r < nRows; ++r) {
            std::string_view a_sv = (r < sa.size()) ? trim(sa[r]) : std::string_view{};
            std::string_view b_sv = (r < sb.size()) ? trim(sb[r]) : std::string_view{};
            bool a_is_num = is_numeric(a_sv);
            bool b_is_num = is_numeric(b_sv);
            if (a_is_num && b_is_num) {
                double a_val = std::stod(std::string(a_sv));
                double b_val = std::stod(std::string(b_sv));
                double diff = std::abs(a_val - b_val);
                sumDiff += diff;
                if (diff > maxDiff) maxDiff = diff;
                if (diff <= tol) ++matched;
            } else if (!a_is_num && !b_is_num) {
                if (a_sv == b_sv) {
                    ++matched;
                } else if (debug_mismatches < 10) {
                    std::cerr << "[DEBUG] String mismatch at row " << r << ", col " << c
                              << ": '" << a_sv << "' vs '" << b_sv << "'\n";
                    ++debug_mismatches;
                }
            } else {
                if (debug_mismatches < 10) {
                    std::cerr << "[DEBUG] Type mismatch at row " << r << ", col " << c
                              << ": '" << std::string(a_sv) << "' vs '" << std::string(b_sv) << "'\n";
                    ++debug_mismatches;
                }
            }
        }
        cs.withinTolRows = matched;
        cs.maxAbsDiff    = maxDiff;
        cs.sumAbsDiff    = sumDiff;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

static std::string basename(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

static void printSeparator(size_t width = 90) {
    std::cout << std::string(width, '-') << '\n';
}

static void printPairResult(const PairResult& pr, const Header& header, double tol, double minMatchPct) {
    std::cout << "\n";
    printSeparator();
    std::cout << "  " << basename(pr.nameA) << "  vs  " << basename(pr.nameB) << "\n";
    std::cout << "  Rows compared: " << pr.rowsCompared
              << "   Tolerance: " << tol
              << "   Min Match%: " << minMatchPct << "%\n";
    printSeparator();

    const int W = 14;
    std::cout << std::left  << std::setw(12) << "Column"
              << std::right << std::setw(W)  << "MaxAbsDiff"
              << std::setw(W) << "MeanAbsDiff"
              << std::setw(W) << "MatchedRows"
              << std::setw(W) << "Match%"
              << "\n";
    printSeparator();

    bool allGood = true;
    for (size_t c = 0; c < pr.colStats.size(); ++c) {
        const auto& cs = pr.colStats[c];
        std::string col = (c < header.size()) ? header[c] : "col" + std::to_string(c);
        bool diff = (cs.matchPct() < minMatchPct);
        if (diff) allGood = false;

        std::cout << std::left  << std::setw(12) << col
                  << std::right << std::setw(W)  << std::scientific << std::setprecision(4)
                  << cs.maxAbsDiff
                  << std::setw(W) << cs.meanAbsDiff()
                  << std::fixed   << std::setprecision(0)
                  << std::setw(W) << cs.withinTolRows
                  << std::setprecision(4) << std::setw(W) << cs.matchPct()
                  << (diff ? "  <-- DIFF" : "")
                  << "\n";
    }
    std::cout << std::defaultfloat << std::setprecision(6);
    printSeparator();
    std::cout << "  Overall: " << (allGood ? "MATCH" : "MISMATCH") << "\n";
}

static std::string shortname(const std::string& path) {
    std::string s = basename(path);
    if (s.size() > 4 && s.substr(s.size() - 4) == ".csv")
        s = s.substr(0, s.size() - 4);
    auto p2 = s.rfind('_');
    if (p2 != std::string::npos && s.size() - p2 - 1 == 6 &&
        std::all_of(s.begin() + p2 + 1, s.end(), ::isdigit)) {
        s = s.substr(0, p2);
        auto p1 = s.rfind('_');
        if (p1 != std::string::npos && s.size() - p1 - 1 == 8 &&
            std::all_of(s.begin() + p1 + 1, s.end(), ::isdigit))
            s = s.substr(0, p1);
    }
    return s;
}

static void printPairResultSimple(const PairResult& pr, const Header& header, double minMatchPct) {
    bool ok = pr.allMatch(minMatchPct);
    std::cout << std::left << std::setw(30) << shortname(pr.nameA)
              << "  vs  "
              << std::setw(30) << shortname(pr.nameB)
              << "  " << (ok ? "MATCH   " : "MISMATCH");

    std::vector<size_t> diff_cols;
    for (size_t c = 0; c < pr.colStats.size(); ++c)
        if (pr.colStats[c].matchPct() < 100.0 - 1e-6)
            diff_cols.push_back(c);

    if (diff_cols.empty()) {
        std::cout << "  all=100.00%";
    } else {
        for (size_t c : diff_cols) {
            const auto& cs = pr.colStats[c];
            std::string col = (c < header.size()) ? header[c] : "col" + std::to_string(c);
            std::cout << "  " << col << "="
                      << std::fixed << std::setprecision(2) << cs.matchPct() << "%";
            if (cs.maxAbsDiff > 0)
                std::cout << "(max=" << std::scientific << std::setprecision(2)
                          << cs.maxAbsDiff << ")";
        }
    }
    std::cout << std::defaultfloat << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    double tol         = 0.001;
    double minMatchPct = 98.0;
    bool   simple      = false;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tol" || arg == "-t") {
            if (i + 1 >= argc) { std::cerr << "Error: --tol requires a value\n"; return 1; }
            tol = std::stod(argv[++i]);
        } else if (arg == "--match" || arg == "-m") {
            if (i + 1 >= argc) { std::cerr << "Error: --match requires a value\n"; return 1; }
            minMatchPct = std::stod(argv[++i]);
        } else if (arg == "--simple" || arg == "-s") {
            simple = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--tol <val>] [--match <pct>] [--simple] file1.csv file2.csv [...]\n"
                      << "  --tol,    -t  Absolute diff tolerance per cell (default: 0.001)\n"
                      << "  --match,  -m  Minimum Match% to consider a column passing (default: 98.0)\n"
                      << "  --simple, -s  One line per pair instead of full per-column table\n";
            return 0;
        } else {
            files.push_back(arg);
        }
    }

    if (files.size() < 2) {
        std::cerr << "Error: need at least 2 CSV files to compare.\n"
                  << "Usage: " << argv[0]
                  << " [--tol <tolerance>] [--simple] file1.csv file2.csv [file3.csv ...]\n";
        return 1;
    }

    // -------------------------------------------------------------------------
    // Load all files in parallel (each file internally uses parallel chunk parsing)
    // -------------------------------------------------------------------------
    std::vector<std::future<CsvFile>> loadFutures;
    loadFutures.reserve(files.size());
    for (const auto& path : files)
        loadFutures.push_back(std::async(std::launch::async, loadCsv, path));

    std::vector<CsvFile> csvs;
    csvs.reserve(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
        try {
            csvs.push_back(loadFutures[i].get());
            if (!simple)
                std::cout << "Loaded " << files[i] << ": "
                          << csvs.back().numRows << " rows, "
                          << csvs.back().header.size() << " cols\n";
        } catch (const std::exception& e) {
            std::cerr << "\nError: " << e.what() << "\n";
            return 1;
        }
    }

    // Validate column consistency
    const Header& refHeader = csvs[0].header;
    for (size_t i = 1; i < csvs.size(); ++i) {
        if (csvs[i].header.size() != refHeader.size()) {
            std::cerr << "[warn] " << csvs[i].path << " has "
                      << csvs[i].header.size() << " columns vs "
                      << refHeader.size() << " in " << csvs[0].path
                      << " — results may be partial\n";
        }
    }

    // -------------------------------------------------------------------------
    // Compare every pair in parallel, collect results in order
    // -------------------------------------------------------------------------
    std::vector<std::pair<size_t, size_t>> pairs;
    for (size_t i = 0; i < csvs.size(); ++i)
        for (size_t j = i + 1; j < csvs.size(); ++j)
            pairs.push_back({i, j});

    std::vector<std::future<PairResult>> cmpFutures;
    cmpFutures.reserve(pairs.size());
    for (auto [i, j] : pairs)
        cmpFutures.push_back(
            std::async(std::launch::async, comparePair,
                       std::cref(csvs[i]), std::cref(csvs[j]), tol));

    std::vector<PairResult> results;
    results.reserve(pairs.size());
    for (auto& fut : cmpFutures) {
        results.push_back(fut.get());
        if (simple)
            printPairResultSimple(results.back(), refHeader, minMatchPct);
        else
            printPairResult(results.back(), refHeader, tol, minMatchPct);
    }

    // -------------------------------------------------------------------------
    // Final summary
    // -------------------------------------------------------------------------
    size_t matched = 0;
    for (const auto& pr : results) { if (pr.allMatch(minMatchPct)) ++matched; }

    if (simple) {
        std::cout << matched << "/" << results.size()
                  << " pairs match  (tol=" << tol << "  min-match=" << minMatchPct << "%)\n";
    } else {
        std::cout << "\n";
        printSeparator();
        std::cout << "  SUMMARY  tol=" << tol << "  min-match=" << minMatchPct << "%\n";
        printSeparator();
        for (const auto& pr : results) {
            bool ok = pr.allMatch(minMatchPct);
            std::cout << "  " << std::left << std::setw(35) << basename(pr.nameA)
                      << "  vs  "
                      << std::setw(35) << basename(pr.nameB)
                      << "  =>  " << (ok ? "MATCH" : "MISMATCH") << "\n";
        }
        printSeparator();
        std::cout << "  " << matched << " / " << results.size() << " pairs match.\n\n";
    }

    return (matched == results.size()) ? 0 : 1;
}
