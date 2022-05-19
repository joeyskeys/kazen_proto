#pragma once

#include <algorithm>
#include <vector>

// Basically duplicated code from Nori

struct DiscrectPDF {
    DiscrectPDF(size_t n_entries = 0) {
        m_cdf.reserve(n_entries + 1);
        m_cdf.clear();
        m_cdf.push_back(0.f);
        m_normalized = false;
    }

    void reserve(size_t n_entries) {
        m_cdf.reserve(n_entries + 1);
    }

    void append(float pdf_val) {
        m_cdf.push_back(m_cdf[m_cdf.size() - 1] + pdf_val);
    }

    size_t size() const {
        return m_cdf.size() - 1;
    }

    float operator[](size_t entry) const {
        return m_cdf[entry + 1] - m_cdf[entry];
    }

    size_t sample(float sp) const {
        std::vector<float>::const_iterator entry =
            std::lower_bound(m_cdf.begin(), m_cdf.end(), sp);
        size_t idx = (size_t) std::max(static_cast<long>(0), entry - m_cdf.begin() - 1);
        return std::min(idx, m_cdf.size() - 2);
    }

    size_t sample(float sp, float& pdf) const {
        size_t idx = sample(sp);
        pdf = operator[](idx);
        return idx;
    }

    float normalize() {
        auto sum = m_cdf[m_cdf.size() - 1];
        if (sum > 0) {
            auto normalization = 1.f / sum;
            for (size_t i = 1; i < m_cdf.size(); ++i)
                m_cdf[i] *= normalization;
            m_cdf[m_cdf.size() - 1] = 1.f;
        }
        return sum;
    }

private:
    std::vector<float> m_cdf;
    bool m_normalized;
};