/* 
 * code in this file was taken from TEMPEST (https://github.com/PrangerStefan/TempestSynthesis)
 */

#include "SparseSmgRpatlHelper.h"

#include <queue>
#include <cmath>
#include <cassert>

#include <storm/environment/solver/GameSolverEnvironment.h>
#include <storm/environment/Environment.h>
#include <storm/environment/solver/MinMaxSolverEnvironment.h>
#include <storm/utility/vector.h>
#include <storm/storage/MaximalEndComponentDecomposition.h>

#include "internal/GameViHelper.h"
#include "internal/Multiplier.h"
#include "GameMaximalEndComponentDecomposition.h"

namespace synthesis {

    void setClippedStatesOfCoalition(storm::storage::BitVector *vector, storm::storage::BitVector relevantStates, storm::storage::BitVector statesOfCoalition) {
        auto clippedStatesCounter = 0;
        for(uint i = 0; i < relevantStates.size(); i++) {
            if(relevantStates.get(i)) {
                vector->set(clippedStatesCounter, statesOfCoalition[i]);
                clippedStatesCounter++;
            }
        }
    }

    bool epsilonGreaterOrEqual(double x, double y, double eps=1e-6) {
        return (x>=y) || (fabs(x - y) <= eps);
    }

    template<typename ValueType>
    SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeUntilProbabilities(storm::Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, storm::modelchecker::ModelCheckerHint const& hint) {
        auto solverEnv = env;
        solverEnv.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration, false);

        // Relevant states are those states which are phiStates and not PsiStates.
        storm::storage::BitVector relevantStates = phiStates & ~psiStates;

        // Initialize the x vector and solution vector result.
        std::vector<ValueType> x = std::vector<ValueType>(relevantStates.getNumberOfSetBits(), storm::utility::zero<ValueType>());
        std::vector<ValueType> result = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
        std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, psiStates);
        std::vector<ValueType> constrainedChoiceValues = std::vector<ValueType>(b.size(), storm::utility::zero<ValueType>());
        std::unique_ptr<storm::storage::Scheduler<ValueType>> scheduler;

        storm::storage::BitVector clippedStatesOfCoalition(relevantStates.getNumberOfSetBits());
        synthesis::setClippedStatesOfCoalition(&clippedStatesOfCoalition, relevantStates, statesOfCoalition);

        auto transposed_matrix = storm::storage::SparseMatrix<ValueType>(transitionMatrix.transpose(true));
        auto endComponentDecomposition = storm::storage::MaximalEndComponentDecomposition<ValueType>(transitionMatrix, transposed_matrix);

        // Fill up the result vector with with 1s for psi states
        storm::utility::vector::setVectorValues(result, psiStates, storm::utility::one<ValueType>());

        if(!relevantStates.empty()) {
            // Reduce the matrix to relevant states.
            storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, relevantStates, relevantStates, false);
            // Create GameViHelper for computations.
            synthesis::GameViHelper<ValueType> viHelper(submatrix, clippedStatesOfCoalition);
            if (produceScheduler) {
                viHelper.setProduceScheduler(true);
            }
            viHelper.performValueIteration(env, x, b, goal.direction(), constrainedChoiceValues);

            // Fill up the constrainedChoice Values to full size.
            viHelper.fillChoiceValuesVector(constrainedChoiceValues, relevantStates, transitionMatrix.getRowGroupIndices());

            // Fill up the result vector with the values of x for the relevant states, with 1s for psi states
            storm::utility::vector::setVectorValues(result, relevantStates, x);

            // if produceScheduler is true, produce scheduler based on the final values from value iteration
            if (produceScheduler) {
                bool maximize = !goal.minimize();
                uint64_t stateCount = transitionMatrix.getRowGroupCount();
                std::vector<uint64_t> optimalChoices(stateCount,0);
                storm::storage::BitVector optimalChoiceSet(stateCount);

                for (auto const& mec : endComponentDecomposition) {
                    std::vector<uint64_t> statesInMEC;
                    std::queue<uint64_t> BFSqueue;
                    // for each MEC compute which max value choices lead to an exit of the MEC
                    for (auto const& stateActions : mec) {
                        uint64_t const& state = stateActions.first;
                        statesInMEC.push_back(state);
                        uint64_t stateChoiceIndex = 0;
                        for (uint64_t row = transitionMatrix.getRowGroupIndices()[state], endRow = transitionMatrix.getRowGroupIndices()[state + 1]; row < endRow; ++row) {
                            // check if the choice belongs to MEC, if not then this choice is the best exit choice for the MEC
                            // the states with such choices will be used as the initial states for backwards BFS
                            if ((!statesOfCoalition.get(state)) && (maximize ? epsilonGreaterOrEqual(constrainedChoiceValues[row], result[state]) : epsilonGreaterOrEqual(result[state], constrainedChoiceValues[row])) && (stateActions.second.find(row) == stateActions.second.end())) {
                                BFSqueue.push(state);
                                optimalChoices[state] = stateChoiceIndex;
                                optimalChoiceSet.set(state);
                                break;
                            } else if ((statesOfCoalition.get(state)) && (maximize ? epsilonGreaterOrEqual(result[state], constrainedChoiceValues[row]) : epsilonGreaterOrEqual(constrainedChoiceValues[row], result[state])) && (stateActions.second.find(row) == stateActions.second.end())) {
                                BFSqueue.push(state);
                                optimalChoices[state] = stateChoiceIndex;
                                optimalChoiceSet.set(state);
                                break;
                            }
                            stateChoiceIndex++;
                        }
                    }

                    // perform BFS on the transposed matrix to identify actions that lead to an exit of MEC
                    while (!BFSqueue.empty()) {
                        auto currentState = BFSqueue.front();
                        BFSqueue.pop();
                        auto transposedRow = transposed_matrix.getRow(currentState);
                        for (auto const &entry : transposedRow) {
                            auto preState = entry.getColumn();
                            auto preStateIt = std::find(statesInMEC.begin(), statesInMEC.end(), preState);
                            if ((preStateIt != statesInMEC.end()) && (!optimalChoiceSet.get(preState))) {
                                uint64_t stateChoiceIndex = 0;
                                bool choiceFound = false;
                                for (uint64_t row = transitionMatrix.getRowGroupIndices()[preState], endRow = transitionMatrix.getRowGroupIndices()[preState + 1]; row < endRow; ++row) {
                                    if ((!statesOfCoalition.get(preState)) && (maximize ? epsilonGreaterOrEqual(constrainedChoiceValues[row], result[preState]) : epsilonGreaterOrEqual(result[preState], constrainedChoiceValues[row]))) {
                                        for (auto const &preStateEntry : transitionMatrix.getRow(row)) {
                                            if (preStateEntry.getColumn() == currentState) {
                                                BFSqueue.push(preState);
                                                optimalChoices[preState] = stateChoiceIndex;
                                                optimalChoiceSet.set(preState);
                                                choiceFound = true;
                                                break;
                                            }
                                        }
                                    } else if ((statesOfCoalition.get(preState)) && (maximize ? epsilonGreaterOrEqual(result[preState], constrainedChoiceValues[row]) : epsilonGreaterOrEqual(constrainedChoiceValues[row], result[preState]))) {
                                        for (auto const &preStateEntry : transitionMatrix.getRow(row)) {
                                            if (preStateEntry.getColumn() == currentState) {
                                                BFSqueue.push(preState);
                                                optimalChoices[preState] = stateChoiceIndex;
                                                optimalChoiceSet.set(preState);
                                                choiceFound = true;
                                                break;
                                            }
                                        }
                                    }
                                    if (choiceFound) {
                                        break;
                                    }
                                    stateChoiceIndex++;
                                }
                            }
                        }
                    }
                }

                // fill in the choices for states outside of MECs
                for (uint64_t state = 0; state < stateCount; state++) {
                    if (optimalChoiceSet.get(state)) {
                        continue;
                    }
                    if (!statesOfCoalition.get(state)) { // not sure why the statesOfCoalition bitvector is flipped
                        uint64_t stateRowIndex = 0;
                        for (auto choice : transitionMatrix.getRowGroupIndices(state)) {
                            if (maximize ? epsilonGreaterOrEqual(constrainedChoiceValues[choice], result[state]) : epsilonGreaterOrEqual(result[state], constrainedChoiceValues[choice])) {
                                optimalChoices[state] = stateRowIndex;
                                optimalChoiceSet.set(state);
                                break;
                            }
                            stateRowIndex++;
                        }
                    } else {
                        uint64_t stateRowIndex = 0;
                        for (auto choice : transitionMatrix.getRowGroupIndices(state)) {
                            if (maximize ? epsilonGreaterOrEqual(result[state], constrainedChoiceValues[choice]) : epsilonGreaterOrEqual(constrainedChoiceValues[choice], result[state])) {
                                optimalChoices[state] = stateRowIndex;
                                optimalChoiceSet.set(state);
                                break;
                            }
                            stateRowIndex++;
                        }
                    }
                }

                //double check if all states have a choice
                for (uint64_t state = 0; state < stateCount; state++) {
                    if (!relevantStates.get(state)) {
                        continue;
                    }
                    assert(optimalChoiceSet.get(state));
                }

                storm::storage::Scheduler<ValueType> tempScheduler(optimalChoices.size());
                for (uint64_t state = 0; state < optimalChoices.size(); ++state) {
                    tempScheduler.setChoice(optimalChoices[state], state);
                }
                scheduler = std::make_unique<storm::storage::Scheduler<ValueType>>(tempScheduler);
            }
        }
        
        return SMGSparseModelCheckingHelperReturnType<ValueType>(std::move(result), std::move(relevantStates), std::move(scheduler), std::move(constrainedChoiceValues));
    }

    template<typename ValueType>
    storm::storage::Scheduler<ValueType> SparseSmgRpatlHelper<ValueType>::expandScheduler(storm::storage::Scheduler<ValueType> scheduler, storm::storage::BitVector psiStates, storm::storage::BitVector notPhiStates) {
        storm::storage::Scheduler<ValueType> completeScheduler(psiStates.size());
        uint_fast64_t maybeStatesCounter = 0;
        uint schedulerSize = psiStates.size();
        for(uint stateCounter = 0; stateCounter < schedulerSize; stateCounter++) {
            // psiStates already fulfill formulae so we can set an arbitrary action
            if(psiStates.get(stateCounter)) {
                completeScheduler.setChoice(0, stateCounter);
            // ~phiStates do not fulfill formulae so we can set an arbitrary action
            } else if(notPhiStates.get(stateCounter)) {
                completeScheduler.setChoice(0, stateCounter);
            } else {
                completeScheduler.setChoice(scheduler.getChoice(maybeStatesCounter), stateCounter);
                maybeStatesCounter++;
            }
        }
        return completeScheduler;
    }

    template<typename ValueType>
    SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeGloballyProbabilities(storm::Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, storm::modelchecker::ModelCheckerHint const& hint) {
        // G psi = not(F(not psi)) = not(true U (not psi))
        // The psiStates are flipped, then the true U part is calculated, at the end the result is flipped again.
        storm::storage::BitVector notPsiStates = ~psiStates;
        statesOfCoalition.complement();

        auto result = computeUntilProbabilities(env, std::move(goal), transitionMatrix, backwardTransitions, storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true), notPsiStates, qualitative, statesOfCoalition, produceScheduler, hint);
        for (auto& element : result.values) {
            element = storm::utility::one<ValueType>() - element;
        }
        for (auto& element : result.choiceValues) {
            element = storm::utility::one<ValueType>() - element;
        }
        return result;
    }

    template<typename ValueType>
    SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeNextProbabilities(storm::Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, storm::modelchecker::ModelCheckerHint const& hint) {
        // Create vector result, bitvector allStates with a true for each state and a vector b for the probability for each state to get to a psi state, choiceValues is to store choices for shielding.
        std::vector<ValueType> result = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
        storm::storage::BitVector allStates = storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true);
        std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(allStates, psiStates);
        std::vector<ValueType> choiceValues = std::vector<ValueType>(transitionMatrix.getRowCount(), storm::utility::zero<ValueType>());
        statesOfCoalition.complement();

        if (produceScheduler) {
            STORM_LOG_WARN("Next formula does not expect that produceScheduler is set to true.");
        }
        // Create a multiplier for reduction.
        auto multiplier = synthesis::MultiplierFactory<ValueType>().create(env, transitionMatrix);
        auto rowGroupIndices = transitionMatrix.getRowGroupIndices();
        rowGroupIndices.erase(rowGroupIndices.begin());
        multiplier->reduce(env, goal.direction(), rowGroupIndices, b, result, nullptr, &statesOfCoalition);
        return SMGSparseModelCheckingHelperReturnType<ValueType>(std::move(result), std::move(allStates), nullptr, std::move(choiceValues));
    }

    template<typename ValueType>
    SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeBoundedGloballyProbabilities(storm::Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, storm::modelchecker::ModelCheckerHint const& hint,uint64_t lowerBound, uint64_t upperBound) {
        // G psi = not(F(not psi)) = not(true U (not psi))
        // The psiStates are flipped, then the true U part is calculated, at the end the result is flipped again.
        storm::storage::BitVector notPsiStates = ~psiStates;
        statesOfCoalition.complement();

        auto result = computeBoundedUntilProbabilities(env, std::move(goal), transitionMatrix, backwardTransitions, storm::storage::BitVector(transitionMatrix.getRowGroupCount(), true), notPsiStates, qualitative, statesOfCoalition, produceScheduler, hint, lowerBound, upperBound, true);
        for (auto& element : result.values) {
            element = storm::utility::one<ValueType>() - element;
        }
        for (auto& element : result.choiceValues) {
            element = storm::utility::one<ValueType>() - element;
        }
        return result;
    }

    template<typename ValueType>
    SMGSparseModelCheckingHelperReturnType<ValueType> SparseSmgRpatlHelper<ValueType>::computeBoundedUntilProbabilities(storm::Environment const& env, storm::solver::SolveGoal<ValueType>&& goal, storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& phiStates, storm::storage::BitVector const& psiStates, bool qualitative, storm::storage::BitVector statesOfCoalition, bool produceScheduler, storm::modelchecker::ModelCheckerHint const& hint,uint64_t lowerBound, uint64_t upperBound, bool computeBoundedGlobally) {
        auto solverEnv = env;
        solverEnv.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration, false);

        // boundedUntil formulas look like:
        // phi U [lowerBound, upperBound] psi
        // --
        // We solve this by look at psiStates, finding phiStates which have paths to psiStates in the given step bounds,
        // then we find all states which have a path to those phiStates in the given lower bound
        // (which states the paths pass before the lower bound does not matter).

        // First initialization of relevantStates between the step bounds.
        storm::storage::BitVector relevantStates = phiStates & ~psiStates;

        // Initializations.
        std::vector<ValueType> x = std::vector<ValueType>(relevantStates.getNumberOfSetBits(), storm::utility::zero<ValueType>());
        std::vector<ValueType> b = transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, psiStates);
        std::vector<ValueType> result = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
        std::vector<ValueType> constrainedChoiceValues = std::vector<ValueType>(transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, psiStates).size(), storm::utility::zero<ValueType>());
        std::unique_ptr<storm::storage::Scheduler<ValueType>> scheduler;

        storm::storage::BitVector clippedStatesOfCoalition(relevantStates.getNumberOfSetBits());
        synthesis::setClippedStatesOfCoalition(&clippedStatesOfCoalition, relevantStates, statesOfCoalition);

        // If there are no relevantStates or the upperBound is 0, no computation is needed.
        if(!relevantStates.empty() && upperBound > 0) {
            // Reduce the matrix to relevant states. - relevant states are all states.
            storm::storage::SparseMatrix<ValueType> submatrix = transitionMatrix.getSubmatrix(true, relevantStates, relevantStates, false);
            // Create GameViHelper for computations.
            synthesis::GameViHelper<ValueType> viHelper(submatrix, clippedStatesOfCoalition);
            if (produceScheduler) {
                viHelper.setProduceScheduler(true);
            }
            // If the lowerBound = 0, value iteration is done until the upperBound.
            if(lowerBound == 0) {
                solverEnv.solver().game().setMaximalNumberOfIterations(upperBound);
                viHelper.performValueIteration(solverEnv, x, b, goal.direction(), constrainedChoiceValues);
            } else {
                // The lowerBound != 0, the first computation between the given bound steps is done.
                solverEnv.solver().game().setMaximalNumberOfIterations(upperBound - lowerBound);
                viHelper.performValueIteration(solverEnv, x, b, goal.direction(), constrainedChoiceValues);

                // Initialization of subResult, fill it with the result of the first computation and 1s for the psiStates in full range.
                std::vector<ValueType> subResult = std::vector<ValueType>(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
                storm::utility::vector::setVectorValues(subResult, relevantStates, x);
                storm::utility::vector::setVectorValues(subResult, psiStates, storm::utility::one<ValueType>());

                // The newPsiStates are those states which can reach the psiStates in the steps between the bounds - the !=0 values in subResult.
                storm::storage::BitVector newPsiStates(subResult.size(), false);
                storm::utility::vector::setNonzeroIndices(subResult, newPsiStates);

                // The relevantStates for the second part of the computation are all states.
                relevantStates = storm::storage::BitVector(phiStates.size(), true);
                submatrix = transitionMatrix.getSubmatrix(true, relevantStates, relevantStates, false);

                // Update the viHelper for the (full-size) submatrix and statesOfCoalition.
                viHelper.updateTransitionMatrix(submatrix);
                viHelper.updateStatesOfCoalition(statesOfCoalition);

                // Reset constrainedChoiceValues and b to 0-vector in the correct dimension.
                constrainedChoiceValues = std::vector<ValueType>(transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, newPsiStates).size(), storm::utility::zero<ValueType>());
                b = std::vector<ValueType>(transitionMatrix.getConstrainedRowGroupSumVector(relevantStates, newPsiStates).size(), storm::utility::zero<ValueType>());

                // The second computation is done between step 0 and the lowerBound
                solverEnv.solver().game().setMaximalNumberOfIterations(lowerBound);
                viHelper.performValueIteration(solverEnv, subResult, b, goal.direction(), constrainedChoiceValues);

                x = subResult;
            }
            viHelper.fillChoiceValuesVector(constrainedChoiceValues, relevantStates, transitionMatrix.getRowGroupIndices());
            if (produceScheduler) {
                scheduler = std::make_unique<storm::storage::Scheduler<ValueType>>(expandScheduler(viHelper.extractScheduler(), relevantStates, ~relevantStates));
            }
            storm::utility::vector::setVectorValues(result, relevantStates, x);
        }
        // In bounded until and bounded eventually formula the psiStates have probability 1 to satisfy the formula,
        // because once reaching a state where psi holds those formulas are satisfied.
        // In bounded globally formulas we cannot set those states to 1 because it is possible to leave a set of safe states after reaching a psiState
        // and in globally the formula has to hold in every time step (between the bounds).
        // e.g. phiState -> phiState -> psiState -> unsafeState
        if(!computeBoundedGlobally){
            storm::utility::vector::setVectorValues(result, psiStates, storm::utility::one<ValueType>());
        }
        return SMGSparseModelCheckingHelperReturnType<ValueType>(std::move(result), std::move(relevantStates), std::move(scheduler), std::move(constrainedChoiceValues));
    }

    template class SparseSmgRpatlHelper<double>;
}
