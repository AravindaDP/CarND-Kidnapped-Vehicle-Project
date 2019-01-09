#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "../src/particle_filter.h"
#include <tuple>
#include <vector>
#include <array>

using ::testing::Eq;

class ParticleTest : public ::testing::Test {
 protected:
  Particle p_ = Particle();
};

class TransformObsTest: public ParticleTest,
                        public ::testing::WithParamInterface<std::tuple<Particle,LandmarkObs,LandmarkObs>> {
 public:
  virtual void SetUp() {
    auto test_data = GetParam();
    p_ = std::get<0>(test_data);
    obs_ = std::get<1>(test_data);
    expected_transform_ = std::get<2>(test_data);
  }

 protected:
  LandmarkObs obs_;
  LandmarkObs expected_transform_;
};

TEST_P(TransformObsTest, ReturnsTransformedObservation_ForGivenObservation) {
  ASSERT_THAT(p_.transformObs(obs_), Eq(expected_transform_));
}

INSTANTIATE_TEST_CASE_P(ParticleTest, TransformObsTest, ::testing::Values(
    std::make_tuple(Particle{0, 4, 5, -M_PI/2, 1, {},{},{}},
                    LandmarkObs{1, 2, 2},
                    LandmarkObs{1, 6, 3}),
    std::make_tuple(Particle{0, 4, 5, -M_PI/2, 1, {},{},{}},
                    LandmarkObs{2, 3, -2},
                    LandmarkObs{2, 2, 2}),
    std::make_tuple(Particle{0, 4, 5, -M_PI/2, 1, {},{},{}},
                    LandmarkObs{3, 0, -4},
                    LandmarkObs{3, 0, 5})));

