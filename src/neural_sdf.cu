/*************************************************************************
    Neural SDF learning demo
    - Learn a 2D signed distance field with a GPU MLP
    - Scene: 3 circles + 2 walls
    Output: gif/neural_sdf.gif
 ************************************************************************/

#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "neural_sdf_nav.cuh"

using namespace std;
using namespace cudabot;

static const char* AVI_PATH = "gif/neural_sdf.avi";
static const char* GIF_PATH = "gif/neural_sdf.gif";

int main() {
    vector<float> train_inputs;
    vector<float> train_targets;
    make_training_set(NeuralSceneKind::DemoWorld, train_inputs, train_targets);

    GpuMLP mlp(NSDF_INPUT_DIM, NSDF_HIDDEN_DIM, NSDF_HIDDEN_LAYERS, NSDF_OUTPUT_DIM);
    mlp.init_random(42);

    vector<float> true_grid = make_true_sdf_grid(NeuralSceneKind::DemoWorld, NSDF_GRID_RES);

    cv::VideoWriter video(
        AVI_PATH,
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        12,
        cv::Size(840, 420));

    if (!video.isOpened()) {
        fprintf(stderr, "Failed to open %s\n", AVI_PATH);
        return 1;
    }

    vector<float> pred_grid = predict_sdf_grid(mlp, NSDF_GRID_RES);
    cv::Mat truth = render_sdf_heatmap(true_grid, NSDF_GRID_RES, NeuralSceneKind::DemoWorld, "True SDF");
    cv::Mat pred = render_sdf_heatmap(pred_grid, NSDF_GRID_RES, NeuralSceneKind::DemoWorld, "MLP Prediction");
    cv::Mat frame;
    cv::hconcat(truth, pred, frame);
    cv::putText(frame, "Epoch 0 / 500", cv::Point(300, 34), cv::FONT_HERSHEY_SIMPLEX,
                0.8, cv::Scalar(255, 255, 255), 2);
    video.write(frame);

    const int chunk_epochs = 25;
    float final_loss = 0.0f;
    auto t0 = chrono::high_resolution_clock::now();
    for (int epoch = chunk_epochs; epoch <= 500; epoch += chunk_epochs) {
        vector<float> losses;
        final_loss = train_neural_sdf(
            mlp, train_inputs, train_targets, chunk_epochs, 256, 0.001f,
            NSDF_ACTIVATION, 2, &losses);

        pred_grid = predict_sdf_grid(mlp, NSDF_GRID_RES);
        truth = render_sdf_heatmap(true_grid, NSDF_GRID_RES, NeuralSceneKind::DemoWorld, "True SDF");
        pred = render_sdf_heatmap(pred_grid, NSDF_GRID_RES, NeuralSceneKind::DemoWorld, "MLP Prediction");
        cv::hconcat(truth, pred, frame);

        char buf[128];
        std::snprintf(buf, sizeof(buf), "Epoch %d / 500  Loss %.4f", epoch, final_loss);
        cv::putText(frame, buf, cv::Point(210, 34), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(255, 255, 255), 2);
        video.write(frame);
        cout << buf << endl;
    }

    auto t1 = chrono::high_resolution_clock::now();
    double train_ms = chrono::duration<double, milli>(t1 - t0).count();
    video.release();

    pred_grid = predict_sdf_grid(mlp, NSDF_GRID_RES);
    double mse = 0.0;
    for (size_t i = 0; i < pred_grid.size(); i++) {
        double d = pred_grid[i] - true_grid[i];
        mse += d * d;
    }
    mse /= pred_grid.size();

    cout << "Training time: " << train_ms << " ms" << endl;
    cout << "Grid MSE: " << mse << endl;
    cout << "Video saved to gif/neural_sdf.avi" << endl;
    convert_avi_to_gif(AVI_PATH, GIF_PATH, 12);
    cout << "GIF saved to gif/neural_sdf.gif" << endl;
    return 0;
}
