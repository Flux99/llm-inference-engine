#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace test_harness {

struct TestCase {
    std::string suite_name;
    std::string test_name;
    std::function<void()> func;
};

inline std::vector<TestCase>& get_registry() {
    static std::vector<TestCase> registry;
    return registry;
}

inline bool register_test(const std::string& suite, const std::string& name, std::function<void()> func) {
    get_registry().push_back({suite, name, std::move(func)});
    return true;
}

inline int run_all_tests() {
    int passed = 0;
    int failed = 0;
    auto& tests = get_registry();

    std::cout << "\n[==========] Running " << tests.size() << " test cases.\n";
    for (const auto& test : tests) {
        std::cout << "[ RUN      ] " << test.suite_name << "." << test.test_name << "\n";
        try {
            test.func();
            std::cout << "[       OK ] " << test.suite_name << "." << test.test_name << "\n";
            passed++;
        } catch (const std::exception& e) {
            std::cout << "[  FAILED  ] " << test.suite_name << "." << test.test_name << " (Exception: " << e.what() << ")\n";
            failed++;
        } catch (...) {
            std::cout << "[  FAILED  ] " << test.suite_name << "." << test.test_name << " (Unknown exception)\n";
            failed++;
        }
    }
    std::cout << "[==========] " << tests.size() << " tests ran.\n";
    std::cout << "[  PASSED  ] " << passed << " tests.\n";
    if (failed > 0) {
        std::cout << "[  FAILED  ] " << failed << " tests.\n";
        return 1;
    }
    return 0;
}

} // namespace test_harness

#define TEST(suite, name) \
    void test_##suite##_##name(); \
    static bool registered_##suite##_##name = ::test_harness::register_test(#suite, #name, test_##suite##_##name); \
    void test_##suite##_##name()

#define EXPECT_EQ(val1, val2) \
    do { \
        if (!((val1) == (val2))) { \
            throw std::runtime_error("EXPECT_EQ failed at line " + std::to_string(__LINE__) + \
                ": " #val1 " != " #val2); \
        } \
    } while (0)

#define EXPECT_FLOAT_EQ(val1, val2) \
    do { \
        if (std::abs((val1) - (val2)) > 1e-5f) { \
            throw std::runtime_error("EXPECT_FLOAT_EQ failed at line " + std::to_string(__LINE__) + \
                ": " #val1 " (" + std::to_string(val1) + ") != " #val2 " (" + std::to_string(val2) + ")"); \
        } \
    } while (0)

#define EXPECT_THROW(stmt, ex_type) \
    do { \
        bool threw = false; \
        try { stmt; } \
        catch (const ex_type&) { threw = true; } \
        catch (...) {} \
        if (!threw) { \
            throw std::runtime_error("EXPECT_THROW failed at line " + std::to_string(__LINE__) + \
                ": Expected exception " #ex_type " was not thrown."); \
        } \
    } while (0)

