#pragma once

#include <set>
#include <limits>

#include <libbase/point.h>

// See https://stackoverflow.com/a/1640399
class FastRandom {
public:
    FastRandom(unsigned long seed=123456789) {
        reset(seed);
    }

    void reset(unsigned long seed=123456789) {
        x = seed;
        y = 362436069;
        z = 521288629;
    }

    // Returns pseudo-random value in range [min; max] (inclusive)
    int next(int min=0, int max=std::numeric_limits<int>::max()) {
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        unsigned long t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return min + (unsigned int) (z % (((unsigned long) max) - min + 1));
    }

    float nextf() {
        return (next() * 2000.0f / std::numeric_limits<int>::max()) - 1000.0f;
    }

    point3uc nextColor() {
        return point3uc(next(0, 255), next(0, 255), next(0, 255));
    }

    std::vector<unsigned int> random_sorted_unique_values(unsigned int min_value, unsigned int max_value, unsigned int count) {
        std::vector<unsigned int> values;

        rassert(min_value + count - 1 <= max_value, 4567891231234, min_value, max_value, count);
        if (min_value + count - 1 == max_value) {
            for (unsigned int i = 0; i < min_value; ++i) {
                values.push_back(min_value + i);
            }
        } else {
            std::set<unsigned int> used;
            unsigned int max_attempts = 100;
            for (unsigned int i = 0; i < count; ++i) {
                unsigned int value = 5467435;
                for (int attempt = 0; attempt < max_attempts; ++attempt) {
                    value = next(min_value, max_value);
                    if (used.count(value) == 0) {
                        used.insert(value);
                        break;
                    } else {
                        rassert(attempt + 1 < max_attempts, 43567843965123, min_value, max_value, count, used.size()); // too much failed attempts
                    }
                }
                rassert(used.size() == i + 1, 435672384312, min_value, max_value, count, used.size());
                values.push_back(value);
            }
            rassert(used.size() == count, 346345243, min_value, max_value, count, used.size());
            values = {used.begin(), used.end()};
            for (int i = 1; i < count; ++i) {
                rassert(values[i - 1] < values[i], 4356234453242);
            }
        }

        rassert(values.size() == count, 45332784931, values.size(), count);
        return values;
    }

private:
    unsigned long x, y, z;
};
