// planner.hpp — choose the best plan for the resources the commander has.
//
// Candidates come from plan_tactics().  The planner greedily adds the action
// with the largest value per resource unit (lazy greedy: earlier gains are
// upper bounds, so only the most promising candidates are re-simulated each
// round) until a resource runs out or nothing helps any more.  Every value is
// measured on a few perturbed scenarios paired with the no-action ensemble,
// so the plan is robust rather than tuned to a single forecast.
//
// Value of a plan = Σ places  weight × protection  +  0.01 × hectares saved,
// where protection is 1 if the fire is kept out for the whole forecast and
// otherwise the delay / 2 h (capped at 1).
#pragma once

#include <string>
#include <vector>

#include "ensemble.hpp"
#include "spread.hpp"
#include "tactics.hpp"

namespace rapid {

struct Resources {
    int aircraft = 2, trucks = 4, dozers = 1;
};

struct PlanTarget {
    std::string id, name;
    double weight = 1;
    std::vector<int> cells;
};

struct PlannerInput {
    const SpreadSolver* solver = nullptr;
    RunConfig cfg;                     // base run: t0, t_end, sources, params
    std::vector<Member> members;       // scenarios to evaluate on (paired with base_arrival)
    const std::vector<std::vector<float>>* base_arrival = nullptr;  // ≥ members.size() no-action runs
    std::vector<PlanTarget> targets;
    std::vector<Intervention> fixed;   // the user's own actions: always in the plan
    std::vector<double> fixed_breach;
    float t_now = 0, t_end = 0;
    double cell_ha = 0.16;
};

struct PlannerOutput {
    int used[3] = {0, 0, 0}, available[3] = {0, 0, 0};  // aircraft, trucks, dozers
    std::vector<std::string> notes;                      // "1 more aircraft would …"
    int simulations = 0;
    double runtime_ms = 0;
};

// Marks cands[i].selected, fills gain / effect / confidence for them.
PlannerOutput choose_plan(const PlannerInput& in, std::vector<Recommendation>& cands, const Resources& budget);

double place_weight(const std::string& kind);
int resource_index(const std::string& resource);  // -1 for none

}  // namespace rapid
