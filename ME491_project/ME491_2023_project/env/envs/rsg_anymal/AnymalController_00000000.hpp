// Copyright (c) 2020 Robotics and Artificial Intelligence Lab, KAIST
//
// Any unauthorized copying, alteration, distribution, transmission,
// performance, display or use of this material is prohibited.
//
// All rights reserved.

#pragma once

#include <set>
#include "../../BasicEigenTypes.hpp"
#include "raisim/World.hpp"

namespace raisim {

/// change the class name and file name ex) AnymalController_00000000 -> AnymalController_STUDENT_ID
class AnymalController_00000000 {

 public:
  inline bool create(raisim::World *world) {
    anymal_ = reinterpret_cast<raisim::ArticulatedSystem *>(world->getObject(name_));
    /// added box
    opposition_ = reinterpret_cast<raisim::Box *>(world->getObject("box"));

      /// get robot data
    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;

    /// initialize containers
    gc_.setZero(gcDim_);
    gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_);
    vTarget_.setZero(gvDim_);
    pTarget12_.setZero(nJoints_);
    boxInit_.setZero(2);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero();
    jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero();
    jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 34;
    actionDim_ = nJoints_;
    actionMean_.setZero(actionDim_);
    actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    actionStd_.setConstant(0.1);

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));
    /// Specifying the body index to be used in terminal state checks.
    /// terminal index for same rule as test
    terminalIndices_.insert(anymal_->getBodyIdx("base"));
    return true;
  }

  inline bool init(raisim::World *world) {
    return true;
  }

  inline bool advance(raisim::World *world, const Eigen::Ref<EigenVec> &action) {
    /// action scaling
    pTarget12_ = action.cast<double>();
    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;
    anymal_->setPdTarget(pTarget_, vTarget_);
    /// added inverse velocity of box to anymal
    auto anyVel = anymal_->getGeneralizedVelocity();
    auto boxVel = anyVel.e().head(3);
    Eigen::Vector3d inverseVelocity(-1.5 * boxVel[0], -1.5 * boxVel[1], 0.0);
    opposition_->setLinearVelocity(inverseVelocity);
    return true;
  }

  inline bool reset(raisim::World *world, double theta) {
    if (playerNum_ == 0) {
      gc_init_.head(3) << 1.5 * std::cos(theta), 1.5 * std::sin(theta), 0.5;
      gc_init_.segment(3, 4) << cos((theta - M_PI) / 2), 0, 0, sin((theta - M_PI) / 2);
    }
    else {
      gc_init_.head(3) << 1.5 * std::cos(theta + M_PI), 1.5 * std::sin(theta + M_PI), 0.5;
      gc_init_.segment(3, 4) << cos(theta / 2), 0, 0, sin(theta / 2);
    }
    anymal_->setState(gc_init_, gv_init_);
    return true;
  }

    /// This function is to calculate the change in distance between previous timesteps.
    /// It returns the positive/ negative change in distance
    inline float calcOpDistDt(raisim::World *world) {
        double distance = (gc_.head(3)-op_.head(3)).norm();
        float reward = 0;
        if (distance < 0.01) {
            distance = 0.01;
        }
        if (distance < previousDistanceO) {
            reward = previousDistanceO-distance;
        }
        if (distance > previousDistanceO) {
            reward = -(distance - previousDistanceO);
        }
        previousDistanceO = distance;
        return reward;
    }

    /// this is a very similar function for rewarding the robot as it moves toward the centre
    inline float calcDistDtReward(raisim::World *world) {
        Eigen::Vector3d center(0,0,0.375);
        double distance = (gc_.head(3)-center).norm();
        float reward = 0;
        if (distance < 0.01) {
            distance = 0.01;
        }
        if (distance < previousDistanceC) {
            reward = previousDistanceC-distance;
        }
        if (distance > previousDistanceC) {
            reward = -(distance - previousDistanceC);
        }
        previousDistanceC = distance;
        return reward;
    }

    /// this function calculates the static distance and returns a positive, inversely proportional score
    inline float calcStaticDistReward(raisim::World *world) {
        Eigen::Vector3d center(0,0,0.375);
        float distance = (gc_.head(3)-center).norm();
        if (distance < 0.01) {
            distance = 0.01;
        }
        return 1/distance;
    }

    /// this function returns the number of contacts with the opponent. In this case the box.
    inline float touchReward(raisim::World *world) {
        float floorReward = 0.0001;
        for (auto &contact: anymal_->getContacts()) {
            if (contact.getPairObjectIndex() == opIndex_ &&
                gc_(2) > 0.25 && gc_(2) < 0.55) {
                floorReward++;
            }
        }
        return floorReward;
    }

    /// this function gives a reward for having a low verticle tilt, to reduce toppling.
    inline float stabliserReward(raisim::World *world) {
        float stableScore = 0;
        auto ori = anymal_->getBaseOrientation();
        raisim::Vec<3> upDirection = {ori(0, 2), ori(1, 2), ori(2, 2)};
        raisim::Vec<3> worldUpDirection = {0, 0, 1};
        double dotProduct = upDirection[0] * worldUpDirection[0] +
                            upDirection[1] * worldUpDirection[1] +
                            upDirection[2] * worldUpDirection[2];
        double normalisedDif = (dotProduct + 1) / 2.0;
        stableScore = normalisedDif;
        return stableScore;
    }

    /// this function provides a reward for having a low centre of mass, again to reduce toppling.
    inline float heightReward(raisim::World *world) {
        double targetHeight = 0.4;
        float heightScore = 0;
        double currentHeight = gc_(2);
        double heightDifference = std::abs(currentHeight - targetHeight);
        if (heightDifference < 0.075) {
            heightScore ++;
        }
        return heightScore;
    }


  inline void updateObservation(raisim::World *world) {
    anymal_->getState(gc_, gv_);
    raisim::Vec<4> quat;
    raisim::Mat<3, 3> rot;
    quat[0] = gc_[3];
    quat[1] = gc_[4];
    quat[2] = gc_[5];
    quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
    /// observe the opposition
    op_ = opposition_->getPosition();
    opIndex_ = opposition_->getIndexInWorld();
    baseIndex_ = anymal_->getBodyIdx("base");

    obDouble_ << gc_[2], /// body pose
    rot.e().row(2).transpose(), /// body orientation
    gc_.tail(12), /// joint angles
    bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
    gv_.tail(12); /// joint velocity
  }

  /// i used an if else here to act similar to curriculem learning.
  /// it scales the rewards depending on the distance from the centre
  /// for example, contacts are more relevant near to the edge
  /// and the distance to the centre is more important as the timer nears 10
  inline void recordReward(Reward *rewards,raisim::World *world, int timer) {
    float dist = gc_.head(2).norm();
    rewards->record("torque", anymal_->getGeneralizedForce().squaredNorm());
    if (dist > 0.5) {
        rewards->record("forwardVel", std::min(4.0, bodyLinearVel_[0])*2.5);
        rewards->record("usefulDist", calcOpDistDt(world)*2.5);
        rewards->record("impactReward", touchReward(world)*2.5);
        rewards->record("staticDist", calcStaticDistReward(world));
        rewards->record("stable1", stabliserReward(world));
        rewards->record("stable2", heightReward(world));
    } else {
        rewards->record("forwardVel", std::min(4.0, bodyLinearVel_[0]));
        rewards->record("usefulDist", calcDistDtReward(world));
        rewards->record("impactReward", touchReward(world));
        rewards->record("staticDist", calcStaticDistReward(world)*timer);
        rewards->record("stable1", stabliserReward(world)*2.5);
        rewards->record("stable2", heightReward(world)*2.5);
    }
  }

  inline const Eigen::VectorXd &getObservation() {
    return obDouble_;
  }

  void setName(const std::string &name) {
    name_ = name;
  }

  void setOpponentName(const std::string &name) {
    opponentName_ = name;
  }

  void setPlayerNum(const int &playerNum) {
    playerNum_ = playerNum;
  }

    inline bool isTerminalState(raisim::World *world) {
        return false;
    }


  inline int getObDim() {
    return obDim_;
  }

  inline int getActionDim() {
    return actionDim_;
  }


 private:
  std::string name_, opponentName_;
  int gcDim_, gvDim_, nJoints_, playerNum_ = 0;
  int opIndex_, baseIndex_;
  raisim::ArticulatedSystem *anymal_;
  raisim::Box *opposition_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, op_;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_, boxInit_;
  std::set<size_t> footIndices_;
  std::set<size_t> terminalIndices_;
  int obDim_ = 0, actionDim_ = 0;
  double forwardVelRewardCoeff_ = 0.;
  double torqueRewardCoeff_ = 0.;
  double distanceToCenterRewardCoeff_ = 0.;
  double distanceToOpponentRewardCoeff_ = 0.;
  double previousDistanceC = 0.8;
  double previousDistanceO = 1.6;


};
}