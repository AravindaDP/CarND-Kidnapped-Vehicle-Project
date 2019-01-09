#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../src/particle_filter.h"
#include <tuple>
#include <vector>
#include <array>

using ::testing::ElementsAreArray;
using ::testing::Eq;

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