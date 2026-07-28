/*
 * Contact-rich Diff-MPPI evaluated against an independent MuJoCo true plant.
 *
 * The controller and gradients retain the nominal smooth CUDA model from
 * benchmark_diff_mppi_pushing_box.cu. Only the state transition after each
 * selected command is delegated to MuJoCo, so this is a closed-loop
 * sim-to-sim transfer test rather than an open-loop trajectory replay.
 */

#define CUDAROBOTICS_PUSHING_BOX_NO_MAIN
#include "benchmark_diff_mppi_pushing_box.cu"

#include <mujoco/mujoco.h>

#include <memory>
#include <stdexcept>

#ifndef CUDAROBOTICS_SOURCE_DIR
#define CUDAROBOTICS_SOURCE_DIR "."
#endif

class MujocoBoxPlant {
public:
    MujocoBoxPlant(const string& model_path, const BoxScenario& scenario,
                   float friction, float mass_scale,
                   float observation_position_std,
                   float observation_angle_std, int frame_skip, int seed)
        : scenario_(scenario), frame_skip_(frame_skip),
          observation_position_std_(observation_position_std),
          observation_angle_std_(observation_angle_std),
          seed_(static_cast<unsigned int>(seed)), rng_(seed_) {
        char error[1024] = {};
        model_ = mj_loadXML(model_path.c_str(), nullptr, error, sizeof(error));
        if (!model_)
            throw runtime_error("MuJoCo model load failed: " + string(error));
        data_ = mj_makeData(model_);
        if (!data_) {
            mj_deleteModel(model_);
            model_ = nullptr;
            throw runtime_error("MuJoCo data allocation failed");
        }
        box_x_ = joint("box_x");
        box_y_ = joint("box_y");
        box_yaw_ = joint("box_yaw");
        pusher_x_ = joint("pusher_x");
        pusher_y_ = joint("pusher_y");
        actuator_x_ = named(mjOBJ_ACTUATOR, "pusher_x_velocity");
        actuator_y_ = named(mjOBJ_ACTUATOR, "pusher_y_velocity");
        box_geom_ = named(mjOBJ_GEOM, "box_geom");
        box_body_ = named(mjOBJ_BODY, "box");
        pusher_geom_ = named(mjOBJ_GEOM, "pusher_geom");
        obstacle_geom_ = named(mjOBJ_GEOM, "obstacle_geom");
        obstacle_body_ = named(mjOBJ_BODY, "obstacle");
        configure_geometry(friction, mass_scale);
    }

    ~MujocoBoxPlant() {
        if (data_) mj_deleteData(data_);
        if (model_) mj_deleteModel(model_);
    }

    MujocoBoxPlant(const MujocoBoxPlant&) = delete;
    MujocoBoxPlant& operator=(const MujocoBoxPlant&) = delete;

    void reset(float px, float py, float ox, float oy, float yaw) {
        rng_.seed(seed_);
        mj_resetData(model_, data_);
        qpos(pusher_x_) = px;
        qpos(pusher_y_) = py;
        qpos(box_x_) = ox;
        qpos(box_y_) = oy;
        qpos(box_yaw_) = yaw;
        mj_forward(model_, data_);
    }

    void observe(
        float& px, float& py, float& ox, float& oy, float& yaw) {
        if (observation_position_std_ > 0.0f) {
            normal_distribution<float> noise(0.0f, observation_position_std_);
            px += noise(rng_);
            py += noise(rng_);
            ox += noise(rng_);
            oy += noise(rng_);
        }
        if (observation_angle_std_ > 0.0f) {
            normal_distribution<float> noise(0.0f, observation_angle_std_);
            yaw = wrapf(yaw + noise(rng_));
        }
    }

    void step(float ux, float uy,
              float& px, float& py, float& ox, float& oy, float& yaw,
              float& vx, float& vy, float& angular_velocity) {
        data_->ctrl[actuator_x_] = clampf_local(ux, -2.0f, 2.0f);
        data_->ctrl[actuator_y_] = clampf_local(uy, -2.0f, 2.0f);
        for (int index = 0; index < frame_skip_; ++index)
            mj_step(model_, data_);
        px = static_cast<float>(qpos(pusher_x_));
        py = static_cast<float>(qpos(pusher_y_));
        ox = static_cast<float>(qpos(box_x_));
        oy = static_cast<float>(qpos(box_y_));
        yaw = static_cast<float>(qpos(box_yaw_));
        vx = static_cast<float>(qvel(box_x_));
        vy = static_cast<float>(qvel(box_y_));
        angular_velocity = static_cast<float>(qvel(box_yaw_));
    }

private:
    int named(mjtObj type, const char* name) const {
        int identifier = mj_name2id(model_, type, name);
        if (identifier < 0)
            throw runtime_error("MuJoCo model is missing " + string(name));
        return identifier;
    }

    int joint(const char* name) const {
        return named(mjOBJ_JOINT, name);
    }

    mjtNum& qpos(int joint_id) {
        return data_->qpos[model_->jnt_qposadr[joint_id]];
    }

    const mjtNum& qpos(int joint_id) const {
        return data_->qpos[model_->jnt_qposadr[joint_id]];
    }

    const mjtNum& qvel(int joint_id) const {
        return data_->qvel[model_->jnt_dofadr[joint_id]];
    }

    void set_friction(int geom, float friction) {
        model_->geom_friction[3 * geom + 0] = friction;
        model_->geom_friction[3 * geom + 1] = 0.01;
        model_->geom_friction[3 * geom + 2] = 0.001;
    }

    void configure_geometry(float friction, float mass_scale) {
        model_->geom_size[3 * box_geom_ + 0] = scenario_.params.hx;
        model_->geom_size[3 * box_geom_ + 1] = scenario_.params.hy;
        model_->geom_size[3 * pusher_geom_ + 0] = scenario_.params.push_r;
        set_friction(box_geom_, friction);
        set_friction(pusher_geom_, friction);
        const mjtNum mass = static_cast<mjtNum>(mass_scale);
        const mjtNum hx = static_cast<mjtNum>(scenario_.params.hx);
        const mjtNum hy = static_cast<mjtNum>(scenario_.params.hy);
        const mjtNum hz = model_->geom_size[3 * box_geom_ + 2];
        model_->body_mass[box_body_] = mass;
        model_->body_inertia[3 * box_body_ + 0] =
            mass * (hy * hy + hz * hz) / 3.0;
        model_->body_inertia[3 * box_body_ + 1] =
            mass * (hx * hx + hz * hz) / 3.0;
        model_->body_inertia[3 * box_body_ + 2] =
            mass * (hx * hx + hy * hy) / 3.0;
        if (scenario_.params.obstacle_count > 0) {
            const float min_x = scenario_.params.obs_min_x;
            const float min_y = scenario_.params.obs_min_y;
            const float max_x = scenario_.params.obs_max_x;
            const float max_y = scenario_.params.obs_max_y;
            model_->body_pos[3 * obstacle_body_ + 0] = 0.5 * (min_x + max_x);
            model_->body_pos[3 * obstacle_body_ + 1] = 0.5 * (min_y + max_y);
            model_->geom_size[3 * obstacle_geom_ + 0] = 0.5 * (max_x - min_x);
            model_->geom_size[3 * obstacle_geom_ + 1] = 0.5 * (max_y - min_y);
            model_->geom_contype[obstacle_geom_] = 1;
            model_->geom_conaffinity[obstacle_geom_] = 1;
        } else {
            model_->body_pos[3 * obstacle_body_ + 0] = 100.0;
            model_->body_pos[3 * obstacle_body_ + 1] = 100.0;
            model_->geom_contype[obstacle_geom_] = 0;
            model_->geom_conaffinity[obstacle_geom_] = 0;
        }
        mj_setConst(model_, data_);
        mj_forward(model_, data_);
    }

    BoxScenario scenario_;
    int frame_skip_;
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    int box_x_ = -1;
    int box_y_ = -1;
    int box_yaw_ = -1;
    int pusher_x_ = -1;
    int pusher_y_ = -1;
    int actuator_x_ = -1;
    int actuator_y_ = -1;
    int box_geom_ = -1;
    int box_body_ = -1;
    int pusher_geom_ = -1;
    int obstacle_geom_ = -1;
    int obstacle_body_ = -1;
    float observation_position_std_ = 0.0f;
    float observation_angle_std_ = 0.0f;
    unsigned int seed_ = 0;
    mt19937 rng_;
};

static string default_model_path() {
    return string(CUDAROBOTICS_SOURCE_DIR) +
           "/mujoco_models/contact_box_push.xml";
}

static int registered_scenario_index(const string& name) {
    const char* names[] = {
        "box_turn",
        "box_align",
        "box_pivot",
        "box_swivel",
        "box_align_strict",
        "box_align_detour",
        "box_align_contact_loss",
        "box_align_contact_arc",
    };
    for (int index = 0; index < 8; ++index)
        if (name == names[index]) return index;
    throw runtime_error("scenario is missing from the registered seed order");
}

int main(int argc, char** argv) {
    if (argc == 2 && string(argv[1]) == "--engine-info") {
        cout << "{\"engine\":\"MuJoCo\",\"version\":\""
             << mj_versionString() << "\",\"version_number\":"
             << mj_version() << ",\"header_version_number\":"
             << mjVERSION_HEADER << "}" << endl;
        return 0;
    }
    string model_path = default_model_path();
    string csv_path = "build/benchmark_diff_mppi_pushing_box_mujoco.csv";
    vector<string> scenario_names;
    vector<string> planner_names;
    vector<int> k_values;
    int seed_count = 8;
    int seed_offset = 0;
    int horizon = DEFAULT_T;
    int frame_skip = 10;
    float friction = 0.6f;
    float mass_scale = 1.0f;
    float observation_position_std = 0.0f;
    float observation_angle_std = 0.0f;
    for (int index = 1; index < argc; ++index) {
        string argument = argv[index];
        if (argument == "--model" && index + 1 < argc)
            model_path = argv[++index];
        else if (argument == "--csv" && index + 1 < argc)
            csv_path = argv[++index];
        else if (argument == "--scenarios" && index + 1 < argc)
            scenario_names = parse_string_list(argv[++index]);
        else if (argument == "--planners" && index + 1 < argc)
            planner_names = parse_string_list(argv[++index]);
        else if (argument == "--k-values" && index + 1 < argc)
            k_values = parse_int_list(argv[++index]);
        else if (argument == "--seed-count" && index + 1 < argc)
            seed_count = atoi(argv[++index]);
        else if (argument == "--seed-offset" && index + 1 < argc)
            seed_offset = atoi(argv[++index]);
        else if (argument == "--horizon" && index + 1 < argc)
            horizon = atoi(argv[++index]);
        else if (argument == "--frame-skip" && index + 1 < argc)
            frame_skip = atoi(argv[++index]);
        else if (argument == "--friction" && index + 1 < argc)
            friction = static_cast<float>(atof(argv[++index]));
        else if (argument == "--box-mass-scale" && index + 1 < argc)
            mass_scale = static_cast<float>(atof(argv[++index]));
        else if (argument == "--observation-position-std" && index + 1 < argc)
            observation_position_std =
                static_cast<float>(atof(argv[++index]));
        else if (argument == "--observation-angle-std" && index + 1 < argc)
            observation_angle_std =
                static_cast<float>(atof(argv[++index]));
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argument.c_str());
            return 1;
        }
    }
    if (seed_count <= 0 || seed_offset < 0 || horizon < 2 ||
        frame_skip <= 0 || !isfinite(friction) || friction < 0.0f ||
        !isfinite(mass_scale) || mass_scale <= 0.0f ||
        !isfinite(observation_position_std) ||
        observation_position_std < 0.0f ||
        !isfinite(observation_angle_std) || observation_angle_std < 0.0f) {
        fprintf(stderr, "Invalid seed, horizon, physics, or observation value.\n");
        return 1;
    }
    if (k_values.empty()) k_values = {128, 256, 512};

    vector<BoxScenario> all_scenarios = {
        make_box_swivel(),
        make_box_align_strict(),
        make_box_align_detour(),
        make_box_align_contact_loss(),
        make_box_align_contact_arc(),
    };
    vector<BoxScenario> scenarios;
    if (scenario_names.empty()) {
        scenarios = all_scenarios;
    } else {
        for (const string& name : scenario_names) {
            auto found = find_if(
                all_scenarios.begin(), all_scenarios.end(),
                [&](const BoxScenario& scenario) {
                    return scenario.name == name;
                });
            if (found == all_scenarios.end()) {
                fprintf(stderr, "Unknown MuJoCo contact scenario: %s\n", name.c_str());
                return 1;
            }
            scenarios.push_back(*found);
        }
    }
    vector<Variant> all_variants;
    { Variant variant; variant.name = "mppi"; all_variants.push_back(variant); }
    { Variant variant; variant.name = "diff_mppi_3"; variant.grad_steps = 3;
      variant.alpha = 0.010f; all_variants.push_back(variant); }
    { Variant variant; variant.name = "soppi_fast";
      variant.use_soppi_sampling = true; variant.soppi_step_size = 0.05f;
      variant.soppi_bandwidth = 2.0f; variant.soppi_neighbor_count = 112;
      variant.soppi_svgd_iters = 2; variant.grad_steps = 1;
      variant.alpha = 0.010f; all_variants.push_back(variant); }
    vector<Variant> variants;
    if (planner_names.empty()) {
        variants = all_variants;
    } else {
        for (const string& name : planner_names) {
            auto found = find_if(
                all_variants.begin(), all_variants.end(),
                [&](const Variant& variant) { return variant.name == name; });
            if (found == all_variants.end()) {
                fprintf(stderr, "Unknown MuJoCo contact planner: %s\n", name.c_str());
                return 1;
            }
            variants.push_back(*found);
        }
    }

    vector<EpisodeMetrics> rows;
    try {
        for (const BoxScenario& scenario : scenarios) {
            const int scenario_index =
                registered_scenario_index(scenario.name);
            for (int k_samples : k_values) {
                for (const Variant& variant : variants) {
                    for (int local_seed = 0; local_seed < seed_count; ++local_seed) {
                        const int seed_index = seed_offset + local_seed;
                        const int seed =
                            6000 + scenario_index * 100 + seed_index * 7 + k_samples;
                        MujocoBoxPlant plant(
                            model_path, scenario, friction, mass_scale,
                            observation_position_std, observation_angle_std,
                            frame_skip, seed);
                        EpisodeRunner runner(
                            variant, scenario, k_samples, horizon, seed);
                        runner.external_plant_reset =
                            [&](float px, float py, float ox, float oy, float yaw) {
                                plant.reset(px, py, ox, oy, yaw);
                            };
                        runner.external_plant_step =
                            [&](float ux, float uy,
                                float& px, float& py, float& ox, float& oy,
                                float& yaw, float& vx, float& vy, float& w) {
                                plant.step(
                                    ux, uy, px, py, ox, oy, yaw, vx, vy, w);
                            };
                        runner.external_plant_observe =
                            [&](float& px, float& py, float& ox, float& oy,
                                float& yaw) {
                                plant.observe(px, py, ox, oy, yaw);
                            };
                        EpisodeMetrics metrics = runner.run();
                        rows.push_back(metrics);
                        printf(
                            "[mujoco:%s] %s K=%d seed_index=%d success=%d "
                            "pos=%.3f ang=%.3f avg_ms=%.3f\n",
                            scenario.name.c_str(), variant.name.c_str(),
                            k_samples, seed_index, metrics.success,
                            metrics.final_distance, metrics.min_goal_distance,
                            metrics.avg_control_ms);
                    }
                }
            }
        }
    } catch (const exception& error) {
        fprintf(stderr, "MuJoCo contact benchmark failed: %s\n", error.what());
        return 1;
    }
    write_csv(rows, csv_path);
    print_summary(rows);
    cout << "MuJoCo contact CSV saved to " << csv_path << endl;
    return 0;
}
