#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../src/particle_filter.h"
#include <tuple>
#include <vector>
#include <array>

using ::testing::ElementsAreArray;
using ::testing::Eq;
using std::default_random_engine;
using std::normal_distribution;

class ParticleFilterTest : public ::testing::Test {
 protected:
  ParticleFilter pf_ = ParticleFilter(3);
};

class InitTest: public ParticleFilterTest,
                public ::testing::WithParamInterface<std::tuple<ground_truth, std::vector<double>,
                                                                std::vector<Particle>>> {
 public:
  virtual void SetUp() {
    auto test_data = GetParam();
    ground_truth gt = std::get<0>(test_data);
    x_ = gt.x;
    y_ = gt.y;
    theta_ = gt.theta;
    std::vector<double> std_pos = std::get<1>(test_data);
    std::copy(std_pos.begin(), std_pos.end(), std_pos_);
    expected_particles_ = std::get<2>(test_data);
  }

 protected:
  double x_;
  double y_;
  double theta_;
  double std_pos_[3];
  std::vector<Particle> expected_particles_;
};

TEST_P(InitTest, SetsParticles_ForInitialPosition) {
  pf_.init(x_, y_, theta_, std_pos_);
  ASSERT_THAT(pf_.particles, ElementsAreArray(expected_particles_));
}

INSTANTIATE_TEST_CASE_P(ParticleFilterTest, InitTest, ::testing::Values(
  std::make_tuple(ground_truth{4983, 5029, 1.201}, std::vector<double>{2, 2, 0.05},
                  std::vector<Particle>{{0, 4982.76, 5030.37, 1.20266, 1, {}, {}, {}},
                                        {1, 4980.83, 5026.85, 1.23824, 1, {}, {}, {}},
                                        {2, 4983.07, 5029.93, 1.30723, 1, {}, {}, {}}})));

class PredictionTest: public ParticleFilterTest,
                      public ::testing::WithParamInterface<std::tuple<Particle, double, std::vector<double>, control_s,
                                                                      Particle>> {
 public:
  virtual void SetUp() {
    pf_ = ParticleFilter(1);
    double std[3] = {0,0,0};
    pf_.init(0,0,0,std);
    auto test_data = GetParam();
    state_ = std::get<0>(test_data);
    delta_t_ = std::get<1>(test_data);
    std::vector<double> std_pos = std::get<2>(test_data);
    std::copy(std_pos.begin(), std_pos.end(), std_pos_);
    control_s c = std::get<3>(test_data);
    velocity_ = c.velocity;
    yaw_rate_ = c.yawrate;
  expected_particle_ = std::get<4>(test_data);
  }

 protected:
  Particle state_;
  double std_pos_[3];
  double velocity_;
  double yaw_rate_;
  double delta_t_;
  Particle expected_particle_;
};

TEST_P(PredictionTest, UpdatesParticles_GivenControlInputValues) {
  pf_.particles = {state_};
  pf_.prediction(delta_t_, std_pos_, velocity_, yaw_rate_);

  ASSERT_THAT(pf_.particles[0], Eq(expected_particle_));
}

INSTANTIATE_TEST_CASE_P(ParticleFilterTest, PredictionTest, ::testing::Values(
    std::make_tuple(Particle{0, 102, 65, 5*M_PI/8, 1, {}, {}, {}}, 0.1, std::vector<double>{0,0,0}, control_s{110, M_PI/8},
                    Particle{0, 97.59, 75.08, 51*M_PI/80, 1, {}, {}, {}})));

class UpdateWeightsTest: public ParticleFilterTest,
                         public ::testing::WithParamInterface<std::tuple<Particle, double, std::vector<double>,
                                                                         std::vector<LandmarkObs>, Map, double>> {
 public:
  virtual void SetUp() {
    pf_ = ParticleFilter(1);
    double std[3] = {0, 0, 0};
    pf_.init(0, 0, 0, std);
    auto test_data = GetParam();
    state_ = std::get<0>(test_data);
    sensor_range_ = std::get<1>(test_data);
    std::vector<double> std_landmark = std::get<2>(test_data);
    std::copy(std_landmark.begin(), std_landmark.end(), std_landmark_);
    observations_ = std::get<3>(test_data);
    map_landmarks_ = std::get<4>(test_data);
    expected_weight_ = std::get<5>(test_data);
  }

 protected:
  Particle state_;
  double sensor_range_;
  double std_landmark_[2]; 
  std::vector<LandmarkObs> observations_;
  Map map_landmarks_;
  double expected_weight_;
};

TEST_P(UpdateWeightsTest, UpdatesParticleWeights_GivenObservationsAndMapLandmarks) {
  pf_.particles = {state_};
  pf_.updateWeights(sensor_range_, std_landmark_, observations_, map_landmarks_);

  ASSERT_NEAR(pf_.particles[0].weight, expected_weight_, pf_.particles[0].weight/100);
}

INSTANTIATE_TEST_CASE_P(ParticleFilterTest, UpdateWeightsTest, ::testing::Values(
    std::make_tuple(Particle{0, 4, 5, -M_PI/2, 1, {},{},{}}, 100, std::vector<double>{0.3, 0.3},
                    std::vector<LandmarkObs>{{-1, 2, 2},
                                             {-1, 3, -2},
                                             {-1, 0, -4}},
                    Map{{{1, 5, 3},
                         {2, 2, 1},
                         {3, 6, 1},
                         {4, 7, 4},
                         {5, 4, 7}}},
                    4.60E-53)));

class DataAssociationTest : public ParticleFilterTest,
                          public ::testing::WithParamInterface<std::tuple<std::vector<LandmarkObs>,
                                                               std::vector<LandmarkObs>, std::vector<LandmarkObs>>> {
 public:
  virtual void SetUp() {
  auto test_data = GetParam();
  predicted_ = std::get<0>(test_data);
  observations_ = std::get<1>(test_data);
  expected_result_ = std::get<2>(test_data);
  }

protected:
  std::vector<LandmarkObs> predicted_;
  std::vector<LandmarkObs> observations_;
  std::vector<LandmarkObs> expected_result_;
};

TEST_P(DataAssociationTest, UpdatesIDsOfObservations_GivenPredictedLandmarks) {
  pf_.dataAssociation(predicted_, observations_);

  ASSERT_THAT(observations_, ElementsAreArray(expected_result_));
}

INSTANTIATE_TEST_CASE_P(ParticleFilterTest, DataAssociationTest, ::testing::Values(
  std::make_tuple(std::vector<LandmarkObs>{{1, 5, 3},
                                         {2, 2, 1},
                                         {3, 6, 1},
                                         {4, 7, 4},
                                         {5, 4, 7}},
                  std::vector<LandmarkObs>{{1, 6, 3},
                                           {2, 2, 2},
                                           {3, 0, 5}},
                  std::vector<LandmarkObs>{{1, 6, 3},
                                           {2, 2, 2},
                                           {2, 0, 5}})));

TEST_F(ParticleFilterTest, ParticleFilter_PassesProjectRubric_GivenEnoughParticles) {
  // parameters related to grading.
  int time_steps_before_lock_required = 100; // number of time steps before accuracy is checked by grader.
  double max_runtime = 45; // Max allowable runtime to pass [sec]
  double max_translation_error = 1; // Max allowable translation error to pass [m]
  double max_yaw_error = 0.05; // Max allowable yaw error [rad]

  // Start timer.
  int start = clock();

  //Set up parameters here
  double delta_t = 0.1; // Time elapsed between measurements [sec]
  double sensor_range = 50; // Sensor range [m]

  /*
   * Sigmas - just an estimate, usually comes from uncertainty of sensor, but
   * if you used fused data from multiple sensors, it's difficult to find
   * these uncertainties directly.
   */
  double sigma_pos[3] = { 0.3, 0.3, 0.01 }; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
  double sigma_landmark[2] = { 0.3, 0.3 }; // Landmark measurement uncertainty [x [m], y [m]]

  // noise generation
  default_random_engine gen;
  normal_distribution<double> N_x_init(0, sigma_pos[0]);
  normal_distribution<double> N_y_init(0, sigma_pos[1]);
  normal_distribution<double> N_theta_init(0, sigma_pos[2]);
  normal_distribution<double> N_obs_x(0, sigma_landmark[0]);
  normal_distribution<double> N_obs_y(0, sigma_landmark[1]);
  double n_x, n_y, n_theta, n_range, n_heading;
  // Read map data
  Map map;
  ASSERT_TRUE(read_map_data("../data/map_data.txt", map));

  // Read position data
  std::vector<control_s> position_meas;
  ASSERT_TRUE(read_control_data("../data/control_data.txt", position_meas));

  // Read ground truth data
  std::vector<ground_truth> gt;
  ASSERT_TRUE(read_gt_data("../data/gt_data.txt", gt));

  // Run particle filter!
  int num_time_steps = position_meas.size();
  ParticleFilter pf(100);
  double total_error[3] = { 0,0,0 };
  double cum_mean_error[3] = { 0,0,0 };

  for (int i = 0; i < num_time_steps; ++i) {
    //cout << "Time step: " << i << endl;
    // Read in landmark observations for current time step.
    std::ostringstream file;
    file << "../data/observation/observations_" << std::setfill('0') << std::setw(6) << i + 1 << ".txt";

    std::vector<LandmarkObs> observations;
    ASSERT_TRUE(read_landmark_data(file.str(), observations));

    // Initialize particle filter if this is the first time step.
    if (!pf.initialized()) {
      n_x = N_x_init(gen);
      n_y = N_y_init(gen);
      n_theta = N_theta_init(gen);
      pf.init(gt[i].x + n_x, gt[i].y + n_y, gt[i].theta + n_theta, sigma_pos);
    }
    else {
      // Predict the vehicle's next state (noiseless).
      pf.prediction(delta_t, sigma_pos, position_meas[i - 1].velocity, position_meas[i - 1].yawrate);
    }
    // simulate the addition of noise to noiseless observation data.
    std::vector<LandmarkObs> noisy_observations;
    LandmarkObs obs;
    for (int j = 0; j < observations.size(); ++j) {
      n_x = N_obs_x(gen);
      n_y = N_obs_y(gen);
      obs = observations[j];
      obs.x = obs.x + n_x;
      obs.y = obs.y + n_y;
      noisy_observations.push_back(obs);
    }

    // Update the weights and resample
    pf.updateWeights(sensor_range, sigma_landmark, noisy_observations, map);
    pf.resample();

    // Calculate and output the average weighted error of the particle filter over all time steps so far.
    std::vector<Particle> particles = pf.particles;
    int num_particles = particles.size();
    double highest_weight = 0.0;
    Particle best_particle;
    for (int i = 0; i < num_particles; ++i) {
      if (particles[i].weight > highest_weight) {
        highest_weight = particles[i].weight;
        best_particle = particles[i];
      }
    }
    double *avg_error = getError(gt[i].x, gt[i].y, gt[i].theta, best_particle.x, best_particle.y, best_particle.theta);

    for (int j = 0; j < 3; ++j) {
      total_error[j] += avg_error[j];
      cum_mean_error[j] = total_error[j] / (double)(i + 1);
    }

    // Print the cumulative weighted error
    //cout << "Cumulative mean weighted error: x " << cum_mean_error[0] << " y " << cum_mean_error[1] << " yaw " << cum_mean_error[2] << endl;

    // If the error is too high, say so and then exit.
    if (i >= time_steps_before_lock_required) {
      ASSERT_TRUE(cum_mean_error[0] <= max_translation_error);
      ASSERT_TRUE(cum_mean_error[1] <= max_translation_error);
      ASSERT_TRUE(cum_mean_error[2] <= max_yaw_error);
    }
  }

  // Output the runtime for the filter.
  int stop = clock();
  double runtime = (stop - start) / double(CLOCKS_PER_SEC);
  //cout << "Runtime (sec): " << runtime << endl;

  // Print success if accuracy and runtime are sufficient (and this isn't just the starter code).
  ASSERT_TRUE(runtime < max_runtime && pf.initialized());
}