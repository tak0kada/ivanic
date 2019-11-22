#pragma once

#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <cassert>
#include <eigen3/Eigen/Core>

Eigen::Matrix3d RotX(const double x)
{
    Eigen::Matrix3d rotation;
    rotation << 1,           0,            0,
                0, std::cos(x), -std::sin(x),
                0, std::sin(x),  std::cos(x);
    return rotation;
}

Eigen::Matrix3d RotY(const double x)
{
    Eigen::Matrix3d rotation;
    rotation <<  std::cos(x), 0, std::sin(x),
                           0, 1,           0,
                -std::sin(x), 0, std::cos(x);
    return rotation;
}

Eigen::Matrix3d RotZ(const double x)
{
    Eigen::Matrix3d rotation;
    rotation << std::cos(x), -std::sin(x), 0,
                std::sin(x),  std::cos(x), 0,
                          0,            0, 1;
    return rotation;
}

inline Eigen::Matrix3d RotZYZ(const double alpha, const double beta, const double gamma)
{
    return RotZ(gamma) * RotY(beta) * RotZ(alpha);
}

inline double Rsref(const std::vector<Eigen::MatrixXd>& Rs, const int l, const int m, const int mp)
{
    assert(std::abs(m) <= l);
    assert(std::abs(mp) <= l);

    return Rs[l](m + l, mp + l);
}

// use Rsref to avoid calculating l
double Rref(const Eigen::MatrixXd& R, const int m, const int mp)
{
    const int l = (R.cols() - 1) / 2;
    assert(std::abs(m) <= l);
    assert(std::abs(mp) <= l);

    return R(m + l, mp + l);
}

double P(const int i, const std::vector<Eigen::MatrixXd>& Rs, const int l, const int m, const int mp)
{
    assert(i == -1 || i == 0 || i == 1);
    assert(std::abs(m) <= l);
    assert(std::abs(mp) <= l);

    if (mp == l)
    {
        return Rsref(Rs, 1, i, 1) * Rsref(Rs, l - 1, m, l - 1) - Rsref(Rs, 1, i, -1) * Rsref(Rs, l - 1, m, -l + 1);
    }
    else if (mp == -l)
    {
        return Rsref(Rs, 1, i, 1) * Rsref(Rs, l - 1, m, -l + 1) + Rsref(Rs, 1, i, -1) * Rsref(Rs, l - 1, m, l - 1);
    }
    else
    {
        return Rsref(Rs, 1, i, 0) * Rsref(Rs, l - 1, m, mp);
    }
}

double U(const std::vector<Eigen::MatrixXd>& Rs, const int l, const int m, const int mp)
{
    assert(std::abs(m) <= l);
    assert(std::abs(mp) <= l);

    return P(0, Rs, l, m, mp);
}

double V(const std::vector<Eigen::MatrixXd>& Rs, const int l, const int m, const int mp)
{
    assert(std::abs(m) <= l);
    assert(std::abs(mp) <= l);

    if (m == 0)
    {
        return P(1, Rs, l, 1, mp) + P(-1, Rs, l, -1, mp);
    }
    else if (m > 0)
    {
        return P(1, Rs, l, m - 1, mp) * (m == 1 ? std::sqrt(2) : 1) - P(-1, Rs, l, -m + 1, mp) * (m == 1 ? 0 : 1);
    }
    else
    {
        return P(1, Rs, l, m + 1, mp) * (m == -1 ? 0 : 1) + P(-1, Rs, l, -m - 1, mp) * (m == -1 ? std::sqrt(2) : 1);
    }
}

double W(const std::vector<Eigen::MatrixXd>& Rs, const int l, const int m, const int mp)
{
    assert(std::abs(m) <= l);
    assert(std::abs(mp) <= l);

    // m does not equal to 0 when this function is called
    if (m > 0)
    {
        return P(1, Rs, l, m + 1, mp) + P(-1, Rs, l, -m - 1, mp);
    }
    else // m < 0
    {
        return P(1, Rs, l, m - 1, mp) - P(-1, Rs, l, -m + 1, mp);
    }
}

inline double calcR(const std::vector<Eigen::MatrixXd>& Rs, const int l, const int m, const int mp)
{
    assert(std::abs(m) <= l);
    assert(std::abs(mp) <= l);

    const double denom = (l == std::abs(mp)) ? 2*l * (2*l - 1) : (l + mp) * (l - mp);
    const double D = (m == 0 ? 1 : 0);
    const double u_numtr = (l + m) * (l - m);
    const double v_numtr = (1 + D) * (l + std::abs(m) - 1) * (l + std::abs(m));
    const double w_numtr = (l - std::abs(m) - 1) * (l - std::abs(m));

    const double u = std::sqrt(u_numtr / denom);
    const double v = 0.5 * std::sqrt(v_numtr / denom) * (1 - 2*D);
    const double w = -0.5 * std::sqrt(w_numtr / denom) * (1 - D);

    double ret = 0;
    if (std::abs(u) > std::numeric_limits<double>::epsilon())
    {
        ret += u * U(Rs, l, m, mp);
    }
    if (std::abs(v) > std::numeric_limits<double>::epsilon())
    {
        ret += v * V(Rs, l, m, mp);
    }
    if (std::abs(w) > std::numeric_limits<double>::epsilon())
    {
        ret += w * W(Rs, l, m, mp);
    }
    return ret;
}

struct RMat
{
    int lmax;
    double alpha;
    double beta;
    double gamma;
    std::vector<Eigen::MatrixXd> Rs; // block diagonal matrix

    RMat(const int lmax, const double alpha, const double beta, const double gamma);
    std::vector<double> operator*(const std::vector<double>& coef) const;
};

RMat::RMat(const int lmax, const double alpha, const double beta, const double gamma)
: lmax(lmax), alpha(alpha), beta(beta), gamma(gamma)
{
    if (static_cast<int>(Rs.size()) != lmax + 1)
    {
        Rs.resize(lmax + 1);
    }

    // l == 0
    Rs[0] = Eigen::MatrixXd::Identity(1, 1);
    // l == 1
    Eigen::Matrix3d rotation = RotZYZ(alpha, beta, gamma);
    Rs[1].resize(3, 3);
    Rs[1] << rotation(1, 1), rotation(1, 2), rotation(1, 0),
             rotation(2, 1), rotation(2, 2), rotation(2, 0),
             rotation(0, 1), rotation(0, 2), rotation(0, 0);
    // l >= 2
    for (int l = 2; l < lmax + 1; ++l)
    {
        Rs[l].resize(2*l + 1, 2*l + 1);
        for (int m = -l; m < l + 1; ++m)
        {
            for (int mp = -l; mp < l + 1; ++mp)
            {
                // set elem
                Rs[l](m + l, mp + l) = calcR(Rs, l, m, mp);
            }
        }
    }
}

std::vector<double> RMat::operator*(const std::vector<double>& coef) const
{
    assert(coef.size() == std::pow(lmax + 1, 2));

    std::vector<double> ret(coef.size());
    // l == 0
    ret[0] = coef[0];
    // l >= 1
    Eigen::VectorXd tmp;
    for (int l = 1; l < lmax + 1; ++l)
    {
        tmp.resize(2*l + 1);
        for (int m = -l; m < l + 1; ++m)
        {
            tmp[m + l] = coef[m + l + l*l];
        }
        tmp = Rs[l] * tmp; // rotate
        for (int m = -l; m < l + 1; ++m)
        {
            ret[m + l + l*l] = tmp[m + l];
        }
    }

    return ret;
}

std::ostream& operator<<(std::ostream& os, const RMat& mat)
{
    for (int l = 0; l < mat.lmax + 1; ++l)
    {
        os << l << ": \n"
           << mat.Rs[l] << "\n";
    }
    os << "\b";
    return os;
}
