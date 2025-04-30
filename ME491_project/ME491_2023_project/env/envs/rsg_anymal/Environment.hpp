// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
// raisim include
#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

#include "../../Yaml.hpp"
#include "../../BasicEigenTypes.hpp"
#include "../../Reward.hpp"

#include TRAINING_HEADER_FILE_TO_INCLUDE

namespace raisim {

class ENVIRONMENT {

public:

    explicit ENVIRONMENT(const std::string &resourceDir, const Yaml::Node &cfg, bool visualizable) :
            visualizable_(visualizable),box_(1,0.5,0.75,20) {
        /// add objects
        auto *robot = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal_red.urdf");
        robot->setName(PLAYER_NAME);
        controller_.setName(PLAYER_NAME);
        robot->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

        //auto* robot2 = world_.addArticulatedSystem(resourceDir + "/anymal/urdf/anymal.urdf");
        //robot2->setName("bot");
        //controller2_.setName("bot");
        //robot2->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

        world_.addGround();

        /// add box as opponent
        auto *box = world_.addBox(1,0.5,0.75,20);
        box->setName("box");
        box_.setName("box");
        box->setPosition(0,0,2);

        controller_.create(&world_);
        //controller2_.create(&world_);
        READ_YAML(double, simulation_dt_, cfg["simulation_dt"])
        READ_YAML(double, control_dt_, cfg["control_dt"])

        /// Reward coefficients
        rewards_.initializeFromConfigurationFile(cfg["reward"]);

        /// visualize if it is the first environment
        if (visualizable_) {
            server_ = std::make_unique<raisim::RaisimServer>(&world_);
            server_->launchServer();
            server_->focusOn(robot);
            auto cage = server_->addVisualCylinder("cage", 3.0, 0.05);
            cage->setPosition(0, 0, 0);
        }
    }

    void init() {}

    /// added resetting the box position
    void reset() {
        auto theta = uniDist_(gen_) * 2 * M_PI;
        controller_.reset(&world_, theta);
        ///controller2_.reset(&world_, theta+2*pi);
        auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER_NAME));
        auto box = reinterpret_cast<raisim::Box *>(world_.getObject("box"));
        int gcDim = anymal->getGeneralizedCoordinateDim();
        Eigen::VectorXd gc;
        gc.setZero(gcDim);
        gc = anymal->getGeneralizedCoordinate().e();
        box->setPosition(-gc(0),-gc(1),gc(2));
        box->setOrientation(gc(3),gc(4),gc(5),gc(6));
        timer_ = 0;
    }

    /// added a timer counter for terminal distance score
    float step(const Eigen::Ref<EigenVec> &action) {
        timer_ += 1;
        controller_.advance(&world_, action);
        //controller2_.advance(&world_, action);
        for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++) {
            if (server_) server_->lockVisualizationServerMutex();
            world_.integrate();
            if (server_) server_->unlockVisualizationServerMutex();
        }
        controller_.updateObservation(&world_);
        //controller2_.updateObservation(&world_);
        controller_.recordReward(&rewards_,&world_,timer_);
        return rewards_.sum();
    }

    void observe(Eigen::Ref<EigenVec> ob) {
        controller_.updateObservation(&world_);
        //controller2_.updateObservation(&world_);
        ob = controller_.getObservation().cast<float>();
    }

    /// i was planning to implement a second anymal,
    /// but it slowed the performance down a lot
    /*
    bool player2_die() {
        auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER2_NAME));
        int gcDim = anymal->getGeneralizedCoordinateDim();
        Eigen::VectorXd gc;
        gc.setZero(gcDim);
        gc = anymal->getGeneralizedCoordinate().e();
        // base contact with ground
        if (gc(3) < 0.1) {
            return true;
        }
        // get out of the cage
        if (gc.head(2).norm() > 3) {
            return true;
        }
        return false;
    }
    */
    /// i used a different method for base contact with the floor
    /// because the program would crash whenever I checked for ground contact
    bool player1_die() {
        auto anymal = reinterpret_cast<raisim::ArticulatedSystem *>(world_.getObject(PLAYER_NAME));
        int gcDim = anymal->getGeneralizedCoordinateDim();
        Eigen::VectorXd gc;
        gc.setZero(gcDim);
        gc = anymal->getGeneralizedCoordinate().e();
        // base contact with ground
        if (gc(2) < 0.3) {
            return true;
        }
        // get out of the cage
        if (gc.head(2).norm() > 3) {
            return true;
        }
        return false;
    }

    /// this scenario never happens as the box moves toward the centre forcefully
    /// but, if my bot manages it then there is a function to score it
    bool box_die() {
        auto box = reinterpret_cast<raisim::Box *>(world_.getObject("box"));
        Eigen::VectorXd gc;
        gc = box->getPosition();
        // get out of the cage
        if (gc.head(2).norm() > 3) {
            return true;
        }
        return false;
    }
    /// i have imported the terminal conditions from the test environment
    bool isTerminalState(float &terminalReward) {
        if (controller_.isTerminalState(&world_)) {
            terminalReward = terminalRewardCoeff_;
            return true;
        }
        if (player1_die()) {
          terminalReward = terminalRewardCoeff_;
          return true;
        }
        /*
        if (player2_die()) {
            terminalReward = -terminalRewardCoeff_;
            return true;
        }
        */
        if (box_die()) {
            terminalReward = -terminalRewardCoeff_;
            return true;
        }
        if (timer_ > 10 * 100) {
            terminalReward = controller_.calcStaticDistReward(&world_);
            return true;
        }
        terminalReward = 0.f;
        return false;
    }

  void curriculumUpdate() {};

  void close() { if (server_) server_->killServer(); };

  void setSeed(int seed) {};

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_.setTimeStep(dt);
  }
  void setControlTimeStep(double dt) { control_dt_ = dt; }

  int getObDim() { return controller_.getObDim(); }

//  int getObDim2() { return controller2_.getObDim(); }

  int getActionDim() { return controller_.getActionDim(); }

  double getControlTimeStep() { return control_dt_; }

  double getSimulationTimeStep() { return simulation_dt_; }

  raisim::World *getWorld() { return &world_; }

  void turnOffVisualization() { server_->hibernate(); }

  void turnOnVisualization() { server_->wakeup(); }

  void startRecordingVideo(const std::string &videoName) { server_->startRecordingVideo(videoName); }

  void stopRecordingVideo() { server_->stopRecordingVideo(); }

  raisim::Reward& getRewards() { return rewards_; }

 private:
  bool visualizable_ = false;
  int timer_ = 0;
  double terminalRewardCoeff_ = -10.;
  TRAINING_CONTROLLER controller_;
  //_CONTROLLER controller2_;
  raisim::World world_;
  raisim::Reward rewards_;
  raisim::Box box_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::unique_ptr<raisim::RaisimServer> server_;
  thread_local static std::uniform_real_distribution<double> uniDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
thread_local std::uniform_real_distribution<double> raisim::ENVIRONMENT::uniDist_(0., 1.);
}

