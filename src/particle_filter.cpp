/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   * first position (based on estimates of x, y, theta and their uncertainties
   * from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  std::default_random_engine gen;

  // Create normal distributions for x, y and theta
  std::normal_distribution<double> N_x(x, std[0]);
  std::normal_distribution<double> N_y(y, std[1]);
  std::normal_distribution<double> N_theta(theta, std[2]);

  for(int i = 0; i < num_particles; i++){
    Particle particle;
    particle.id = i;
    particle.x = N_x(gen);
    particle.y = N_y(gen);
    particle.theta = N_theta(gen);
    particle.weight = 1;

    particles.push_back(particle);
    weights.push_back(1);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  for(int i = 0; i < num_particles; i++){	
    double new_x;
    double new_y;
    double new_theta;

    if(fabs(yaw_rate) < 1e-6){
      new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
      new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
      new_theta = particles[i].theta;
    }
    else{
      new_x = particles[i].x + velocity/yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      new_y = particles[i].y + velocity/yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      new_theta = particles[i].theta + yaw_rate*delta_t;
    }

    std::normal_distribution<double> N_x(new_x, std_pos[0]);
    std::normal_distribution<double> N_y(new_y, std_pos[1]);
    std::normal_distribution<double> N_theta(new_theta, std_pos[2]);

    particles[i].x = N_x(gen);
    particles[i].y = N_y(gen);
    particles[i].theta = N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   * observed measurement and assign the observed measurement to this 
   * particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  int num_obs = observations.size();
  int num_landmarks = predicted.size();
  for (int i = 0; i < num_obs; ++i) {
    int closest_landmark = 0;
    int min_dist = 999999;
    int curr_dist;
    // Iterate through all landmarks to check which is closest
    for (int j = 0; j < num_landmarks; ++j) {
      // Calculate Euclidean distance
      curr_dist = sqrt(pow(observations[i].x - predicted[j].x, 2)
                       + pow(observations[i].y - predicted[j].y, 2));
      // Compare to min_dist and update if closest
      if (curr_dist < min_dist) {
        min_dist = curr_dist;
        closest_landmark = j;
      }
    }
    // Output the related association information
    //std::cout << "OBS" << observations[i].id << " associated to L"
    //          << predicted[closest_landmark].id << std::endl;

		observations[i].id = predicted[closest_landmark].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   * distribution. You can read more about this distribution here: 
   * https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  int num_obs = observations.size();
  double sensor_range_std = sqrt(pow(std_landmark[0],2) + pow(std_landmark[1],2));
  // Through all particles
  for(int p = 0; p < num_particles; p++){
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    // create vectors hold transformations
    vector<LandmarkObs> trans_observations(num_obs);
    // Iterate through our three observations to transform them
    for(int i = 0; i < num_obs; ++i){
      trans_observations[i] = particles[p].transformObs(observations[i]);
      trans_observations[i].id = 0;
    }

    particles[p].weight = 1.0;
     
    //Predicted observations <- Landmarks within sensor_range
    vector<LandmarkObs> pred_observations;
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++){		
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      double calc_dist = dist(particles[p].x, particles[p].y, landmark_x, landmark_y);
      if(calc_dist < sensor_range + sensor_range_std*2){ //2*sigma for 95% of measurements
        LandmarkObs pred_obs;
        pred_obs.id = map_landmarks.landmark_list[j].id_i;
        pred_obs.x = landmark_x;
        pred_obs.y = landmark_y;
        pred_observations.push_back(pred_obs);
      }
    }

    dataAssociation(pred_observations, trans_observations);

    for(int i = 0; i < num_obs; i++){ 
      int association = trans_observations[i].id;

      if(association!=0){
        double meas_x = trans_observations[i].x;
        double meas_y = trans_observations[i].y;
        double mu_x = map_landmarks.landmark_list[association-1].x_f;
        double mu_y = map_landmarks.landmark_list[association-1].y_f;
        long double multipler = multiv_prob(std_landmark[0], std_landmark[1], meas_x, meas_y, mu_x, mu_y);
        if(multipler > 0){
          particles[p].weight *= multipler;
        }
      }
      associations.push_back(association);
      sense_x.push_back(trans_observations[i].x);
      sense_y.push_back(trans_observations[i].y);
    }

    SetAssociations(particles[p],associations,sense_x,sense_y);
    weights[p] = particles[p].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   * to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
	std::default_random_engine gen;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resample_particles;

	for(int i = 0; i < num_particles; i++){
		resample_particles.push_back(particles[distribution(gen)]);
	}

	particles = resample_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}