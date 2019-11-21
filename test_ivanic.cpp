#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>
// #include <boost/test/tools/detail/tolerance_manip.hpp>

#include "./ivanic.hpp"
#include <iostream>
// #include <ctime>
#include <cmath>
#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

BOOST_AUTO_TEST_CASE(test_ivanic_initialization)
{
    int lmax = 10;
    double alpha = 10;
    double beta = 20;
    double gamma = 30;
    std::printf("Executing RMat initialization (RMat(%d, %f, %f, %f))...\n", lmax, alpha, beta, gamma);

    // std::clock_t start = std::clock();
    auto t_start = std::chrono::high_resolution_clock::now();
    RMat mat(lmax, alpha, beta, gamma);
    auto t_end = std::chrono::high_resolution_clock::now();
    // std::clock_t end = std::clock();

    // std::cout << "CPU time used: " << 1000 * (end - start) / CLOCKS_PER_SEC << " ms" << std::endl;
    std::cout << "      Wall clock time passed: "
              << std::chrono::duration<double, std::milli>(t_end-t_start).count()
              << " msec"
              << std::endl;
}

BOOST_AUTO_TEST_CASE(test_ivanic_value)
{
    std::cout << "Running value correctness test..." << std::endl;

    int lmax = 2;
    double alpha = 10;
    double beta = 20;
    double gamma = 30;
    RMat mat(lmax, alpha, beta, gamma);

    const Eigen::MatrixXd R_eu = RotZ(gamma) * RotY(beta) * RotZ(alpha);
    Eigen::MatrixXd R;
    R.resize(3, 3);
    R << R_eu(1, 1), R_eu(1, 2), R_eu(1, 0),
         R_eu(2, 1), R_eu(2, 2), R_eu(2, 0),
         R_eu(0, 1), R_eu(0, 2), R_eu(0, 0);

    Eigen::MatrixXd ans;
    ans.resize(5,5);
    ans(2, 2) = 0.5 * (3 * std::pow(Rref(R, 0, 0), 2) - 1);
    ans(2, 3) = std::sqrt(3) * Rref(R, 0, 1) * Rref(R, 0, 0);
    ans(2, 1) = std::sqrt(3) * Rref(R, 0, -1) * Rref(R, 0, 0);
    ans(2, 4) = 0.5 * std::sqrt(3) * (std::pow(Rref(R, 0, 1), 2) - std::pow(Rref(R, 0, -1), 2));
    ans(2, 0) = std::sqrt(3) * Rref(R, 0, 1) * Rref(R, 0, -1);
    ans(3, 2) = std::sqrt(3) * Rref(R, 1, 0) * Rref(R, 0, 0);
    ans(3, 3) = Rref(R, 1, 1) * Rref(R, 0, 0) + Rref(R, 1, 0) * Rref(R, 0, 1);
    ans(3, 1) = Rref(R, 1, -1) * Rref(R, 0, 0) + Rref(R, 1, 0) * Rref(R, 0, -1);
    ans(3, 4) = Rref(R, 1, 1) * Rref(R, 0, 1) - Rref(R, 1, -1) * Rref(R, 0, -1);
    ans(3, 0) = Rref(R, 1, 1) * Rref(R, 0, -1) + Rref(R, 1, -1) * Rref(R, 0, 1);
    ans(1, 2) = std::sqrt(3) * Rref(R, -1, 0) * Rref(R, 0, 0);
    ans(1, 3) = Rref(R, -1, 1) * Rref(R, 0, 0) + Rref(R, -1, 0) * Rref(R, 0, 1);
    ans(1, 1) = Rref(R, -1, -1) * Rref(R, 0, 0) + Rref(R, -1, 0) * Rref(R, 0, -1);
    ans(1, 4) = Rref(R, -1, 1) * Rref(R, 0, 1) - Rref(R, -1, -1) * Rref(R, 0, -1);
    ans(1, 0) = Rref(R, -1, 1) * Rref(R, 0, -1) + Rref(R, -1, -1) * Rref(R, 0, 1);
    ans(4, 2) = 0.5 * std::sqrt(3) * (std::pow(Rref(R, 1, 0), 2) - std::pow(Rref(R, -1, 0), 2));
    ans(4, 3) = Rref(R, 1, 1) * Rref(R, 1, 0) - Rref(R, -1, 1) * Rref(R, -1, 0);
    ans(4, 1) = Rref(R, 1, -1) * Rref(R, 1, 0) - Rref(R, -1, -1) * Rref(R, -1, 0);
    ans(4, 4) = 0.5 * (std::pow(Rref(R, 1, 1), 2) - std::pow(Rref(R, 1, -1), 2) - std::pow(Rref(R, -1, 1), 2) + std::pow(Rref(R, -1, -1), 2));
    ans(4, 0) = Rref(R, 1, 1) * Rref(R, 1, -1) - Rref(R, -1, 1) * Rref(R, -1, -1);
    ans(0, 2) = std::sqrt(3) * Rref(R, 1, 0) * Rref(R, -1, 0);
    ans(0, 3) = Rref(R, 1, 1) * Rref(R, -1, 0) + Rref(R, 1, 0) * Rref(R, -1, 1);
    ans(0, 1) = Rref(R, 1, -1) * Rref(R, -1, 0) + Rref(R, 1, 0) * Rref(R, -1, -1);
    ans(0, 4) = Rref(R, 1, 1) * Rref(R, -1, 1) - Rref(R, 1, -1) * Rref(R, -1, -1);
    ans(0, 0) = Rref(R, 1, 1) * Rref(R, -1, -1) + Rref(R, 1, -1) * Rref(R, -1, 1);

    BOOST_ASSERT(mat.Rs[2].isApprox(ans, std::numeric_limits<double>::epsilon()));

    std::cout << "      Rmat(2, 10, 20, 30).Rs[2] == reference" << std::endl;
}
