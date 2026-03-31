#include "autodiff_engine.cuh"
#include <cstdio>
#include <cmath>

using namespace cudabot;

// ===== Test 1: f(x) = sin(x) * x^2 の微分を x=1.0 で検証 =====
bool test_sin_x_squared() {
    printf("[Test 1] f(x) = sin(x) * x^2, f'(x) at x=1.0\n");

    // Autodiff
    Dualf x = Dualf::variable(1.0f);
    Dualf result = sin(x) * x * x;

    float ad_val = result.val;
    float ad_deriv = result.deriv;

    // Numerical differentiation: (f(x+h) - f(x-h)) / (2h)
    double h = 1e-4;
    double xv = 1.0;
    double f_plus = std::sin(xv + h) * (xv + h) * (xv + h);
    double f_minus = std::sin(xv - h) * (xv - h) * (xv - h);
    float num_deriv = static_cast<float>((f_plus - f_minus) / (2.0 * h));

    // Analytical: f'(x) = cos(x)*x^2 + sin(x)*2x
    float anal_val = std::sin(1.0f) * 1.0f * 1.0f;
    float anal_deriv = std::cos(1.0f) * 1.0f * 1.0f + std::sin(1.0f) * 2.0f * 1.0f;

    printf("  Autodiff:   val=%.6f, deriv=%.6f\n", ad_val, ad_deriv);
    printf("  Analytical: val=%.6f, deriv=%.6f\n", anal_val, anal_deriv);
    printf("  Numerical:  deriv=%.6f\n", num_deriv);

    float tol = 1e-3f;
    bool pass = std::fabs(ad_val - anal_val) < tol &&
                std::fabs(ad_deriv - anal_deriv) < tol &&
                std::fabs(ad_deriv - num_deriv) < tol;
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ===== Test 2: Bicycle dynamics Jacobian =====
// state = (x, y, theta, v), control = (accel, steer)
// x' = x + v*cos(theta)*dt
// y' = y + v*sin(theta)*dt
// theta' = theta + v/L*tan(steer)*dt
// v' = v + accel*dt
struct BicycleParams {
    float L = 2.5f;
    float dt = 0.05f;
};

void bicycle_step_dual(Dualf& x, Dualf& y, Dualf& theta, Dualf& v,
                       Dualf accel, Dualf steer, const BicycleParams& p) {
    Dualf x_new = x + v * cos(theta) * p.dt;
    Dualf y_new = y + v * sin(theta) * p.dt;
    Dualf theta_new = theta + v / Dualf::constant(p.L) * tan(steer) * p.dt;
    Dualf v_new = v + accel * p.dt;
    x = x_new;
    y = y_new;
    theta = theta_new;
    v = v_new;
}

bool test_bicycle_jacobian() {
    printf("[Test 2] Bicycle dynamics Jacobian\n");

    float state[4] = {10.0f, 5.0f, 0.3f, 2.0f};  // x, y, theta, v
    float ctrl[2] = {0.5f, 0.1f};                   // accel, steer
    BicycleParams bp;

    // Compute Jacobian: 4x6 (4 outputs, 6 inputs = 4 state + 2 control)
    float J[4][6];

    // For each input variable (6 total), run forward pass with that variable as dual
    for (int var = 0; var < 6; var++) {
        Dualf x_d(state[0]);
        Dualf y_d(state[1]);
        Dualf theta_d(state[2]);
        Dualf v_d(state[3]);
        Dualf a_d(ctrl[0]);
        Dualf s_d(ctrl[1]);

        // Set the variable we're differentiating w.r.t.
        Dualf* vars[6] = {&x_d, &y_d, &theta_d, &v_d, &a_d, &s_d};
        *vars[var] = Dualf::variable(vars[var]->val);

        bicycle_step_dual(x_d, y_d, theta_d, v_d, a_d, s_d, bp);

        J[0][var] = x_d.deriv;
        J[1][var] = y_d.deriv;
        J[2][var] = theta_d.deriv;
        J[3][var] = v_d.deriv;
    }

    // Verify against numerical differentiation
    double h = 1e-4;
    float J_num[4][6];
    float inputs[6] = {state[0], state[1], state[2], state[3], ctrl[0], ctrl[1]};

    for (int var = 0; var < 6; var++) {
        // f(x+h)
        double inp_p[6], inp_m[6];
        for (int k = 0; k < 6; k++) { inp_p[k] = inputs[k]; inp_m[k] = inputs[k]; }
        inp_p[var] += h;
        inp_m[var] -= h;

        auto step = [&](const double* inp, double out[4]) {
            double xx = inp[0], yy = inp[1], th = inp[2], vv = inp[3];
            double aa = inp[4], ss = inp[5];
            out[0] = xx + vv * std::cos(th) * bp.dt;
            out[1] = yy + vv * std::sin(th) * bp.dt;
            out[2] = th + vv / bp.L * std::tan(ss) * bp.dt;
            out[3] = vv + aa * bp.dt;
        };

        double out_p[4], out_m[4];
        step(inp_p, out_p);
        step(inp_m, out_m);

        for (int i = 0; i < 4; i++) {
            J_num[i][var] = static_cast<float>((out_p[i] - out_m[i]) / (2.0 * h));
        }
    }

    bool pass = true;
    float tol = 1e-3f;
    printf("  Jacobian (autodiff vs numerical):\n");
    for (int i = 0; i < 4; i++) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 6; j++) {
            printf("%.4f/%.4f ", J[i][j], J_num[i][j]);
            if (std::fabs(J[i][j] - J_num[i][j]) > tol) pass = false;
        }
        printf("\n");
    }
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ===== Test 3: GPU kernel test =====
__device__ Dualf gpu_test_func(Dualf x) {
    return sin(x) * x * x;
}

__global__ void test_autodiff_kernel(float* results) {
    Dualf x = Dualf::variable(1.0f);
    Dualf r = gpu_test_func(x);
    results[0] = r.val;
    results[1] = r.deriv;

    // Also test other operations
    Dualf a = Dualf::variable(2.0f);
    Dualf b = Dualf::constant(3.0f);
    Dualf c = exp(a) + log(b);
    results[2] = c.val;
    results[3] = c.deriv;

    // Test atan2
    Dualf y_d = Dualf::variable(1.0f);
    Dualf x_d = Dualf::constant(1.0f);
    Dualf at = atan2(y_d, x_d);
    results[4] = at.val;
    results[5] = at.deriv;
}

bool test_gpu_kernel() {
    printf("[Test 3] GPU kernel autodiff test\n");

    float* d_results;
    cudaMalloc(&d_results, 6 * sizeof(float));

    test_autodiff_kernel<<<1, 1>>>(d_results);
    cudaDeviceSynchronize();

    float results[6];
    cudaMemcpy(results, d_results, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    // sin(1)*1^2 = 0.8415
    float expected_val = std::sin(1.0f);
    float expected_deriv = std::cos(1.0f) + 2.0f * std::sin(1.0f);

    // exp(2) + log(3)
    float expected_exp_log_val = std::exp(2.0f) + std::log(3.0f);
    float expected_exp_log_deriv = std::exp(2.0f);  // d/da of exp(a) + log(3)

    // atan2(1, 1) = pi/4, d/dy atan2(y,x) = x/(x^2+y^2) = 1/2
    float expected_atan2_val = std::atan2(1.0f, 1.0f);
    float expected_atan2_deriv = 1.0f / (1.0f + 1.0f + 1e-10f);

    printf("  sin(x)*x^2 at x=1: val=%.4f (exp %.4f), deriv=%.4f (exp %.4f)\n",
           results[0], expected_val, results[1], expected_deriv);
    printf("  exp(a)+log(3) at a=2: val=%.4f (exp %.4f), deriv=%.4f (exp %.4f)\n",
           results[2], expected_exp_log_val, results[3], expected_exp_log_deriv);
    printf("  atan2(y,1) at y=1: val=%.4f (exp %.4f), deriv=%.4f (exp %.4f)\n",
           results[4], expected_atan2_val, results[5], expected_atan2_deriv);

    float tol = 1e-3f;
    bool pass = std::fabs(results[0] - expected_val) < tol &&
                std::fabs(results[1] - expected_deriv) < tol &&
                std::fabs(results[2] - expected_exp_log_val) < tol &&
                std::fabs(results[3] - expected_exp_log_deriv) < tol &&
                std::fabs(results[4] - expected_atan2_val) < tol &&
                std::fabs(results[5] - expected_atan2_deriv) < tol;
    printf("  %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("========================================\n");
    printf("  Autodiff Engine Tests\n");
    printf("========================================\n\n");

    bool all_pass = true;
    all_pass &= test_sin_x_squared();
    all_pass &= test_bicycle_jacobian();
    all_pass &= test_gpu_kernel();

    if (all_pass) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
